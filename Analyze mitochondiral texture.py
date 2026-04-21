import os
import re
import csv
import glob
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from skimage import io, filters, feature, exposure, morphology, color, segmentation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict
import concurrent.futures
import gc


# ==========================================
# IO AND SETUP
# ==========================================

def setup_output(base_path, folder_name="Output masks", zero_pad=True):
    """creates output directory and subdirectories for a 96-well plate."""
    output_dir = os.path.join(base_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    rows = "ABCDEFGH"
    cols = range(1, 13)
    well_paths = {}
    
    for row in rows:
        for col in cols:
            well_id = f"{row}{col:02d}" if zero_pad else f"{row}{col}"
            well_path = os.path.join(output_dir, well_id)
            os.makedirs(well_path, exist_ok=True)
            well_paths[well_id] = well_path
            
    print(f"Generated 96-well plate structure in: {output_dir}")
    return output_dir, well_paths


# ==========================================
# TECAN PARSER
# ==========================================

def parse_tecan_d300e_xml(file_path):
    """
    parses a Tecan D300e .dtt XML file.
    tracks where the software physically dispensed the fluids and vehicles after randomization.
    returns a dictionary mapping physical well coordinates to treatment strings.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    protocol = root.find('Protocol')

    unit_node = protocol.find('MolarityConcentrationUnit')
    unit = unit_node.text if unit_node is not None else "uM"

    fluid_map = {}
    for fluid in protocol.find('Fluids').findall('Fluid'):
        fluid_id = fluid.get('ID')
        name = fluid.find('Name').text
        fluid_map[fluid_id] = name

    plate = protocol.find('.//Plates/Plate')

    final_destination_map = {}
    randomize_node = plate.find('Randomize')
    if randomize_node is not None:
        for r_well in randomize_node.findall('Well'):
            physical_pos = (int(r_well.get('R')), int(r_well.get('C')))
            logical_pos = (int(r_well.get('ToR')), int(r_well.get('ToC')))
            final_destination_map[logical_pos] = physical_pos

    well_treatments = defaultdict(list)
    
    for well in plate.find('Wells').findall('Well'):
        recipe_r = int(well.get('Row'))
        recipe_c = int(well.get('Col'))
        
        phys_pos = final_destination_map.get((recipe_r, recipe_c), (recipe_r, recipe_c))
        
        for fluid in well.findall('Fluid'):
            f_id = fluid.get('ID')
            concentration = fluid.text
            drug_name = fluid_map.get(f_id, "Unknown")
            well_treatments[phys_pos].append(f"{drug_name} {concentration}{unit}")

    vehicle_wells = set()
    backfill_node = protocol.find('.//Backfills')
    if backfill_node is not None:
        for well in backfill_node.findall('.//Well'):
            recipe_r = int(well.get('R'))
            recipe_c = int(well.get('C'))
            
            phys_pos = final_destination_map.get((recipe_r, recipe_c), (recipe_r, recipe_c))
            vehicle_wells.add(phys_pos)

    treatment_output = {}
    all_used_wells = set(well_treatments.keys()).union(vehicle_wells)

    for phys_r, phys_c in all_used_wells:
        row_letter = chr(65 + phys_r) 
        col_num = phys_c + 1
        well_name = f"{row_letter}{col_num:02d}"
        
        if (phys_r, phys_c) in well_treatments and well_treatments[(phys_r, phys_c)]:
            treatment_output[well_name] = " + ".join(well_treatments[(phys_r, phys_c)])
        else:
            treatment_output[well_name] = "Vehicle"

    return treatment_output


# ==========================================
# IMAGE PROCESSING FUNCTIONS
# ==========================================

def load_as_float(image_path):
    """loads image and forces float for safe math operations"""
    if not os.path.exists(image_path):
        return None
        
    img = io.imread(image_path, as_gray=True)
    if np.max(img) == 0:
        return img
        
    if img.dtype not in [np.float64, np.float32]:
        return img.astype(np.float32) / np.max(img)
    return img

def segment_nuclei(img, min_nucleus_area=500, use_watershed=False):
    """handles hoechst segmentation to find nuclei markers"""
    if img is None or np.max(img) == 0: 
        return None, 0
        
    img_smoothed = filters.gaussian(img, sigma=2.0)
    thresh = filters.threshold_otsu(img_smoothed)
    mask = img_smoothed > thresh
    
    mask_cleaned = morphology.remove_small_objects(mask, max_size=min_nucleus_area)
    
    if use_watershed:
        distance = ndi.distance_transform_edt(mask_cleaned)
        coords = feature.peak_local_max(distance, min_distance=7, labels=mask_cleaned)
        marker_mask = np.zeros(distance.shape, dtype=bool)
        marker_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(marker_mask)
        nuclei_labels = segmentation.watershed(-distance, markers, mask=mask_cleaned)
        num_cells = np.max(nuclei_labels)
    else:
        nuclei_labels, num_cells = ndi.label(mask_cleaned)

    return nuclei_labels, num_cells

def define_cell_territories(img_hoechst, nuclei_labels, mask_mtg, mask_tmrm, out_folder, base_name, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3, save_qc=True):
    """uses a topographical watershed to organically wrap around the mitochondrial networks."""
    if nuclei_labels is None:
        return None
        
    mito_footprint = np.zeros(nuclei_labels.shape, dtype=bool)
    if mask_mtg is not None:
        mito_footprint |= mask_mtg
    if mask_tmrm is not None:
        mito_footprint |= mask_tmrm
        
    flow_channel = morphology.binary_dilation(mito_footprint, morphology.disk(3))
    mito_envelope = morphology.binary_dilation(mito_footprint, morphology.disk(mito_buffer))
    base_cyto = segmentation.expand_labels(nuclei_labels, distance=min_cyto) > 0
    valid_area = mito_envelope | base_cyto
    max_bound = segmentation.expand_labels(nuclei_labels, distance=cyto_expansion) > 0
    flood_mask = valid_area & max_bound
    
    dist_from_mito = ndi.distance_transform_edt(~flow_channel)
    dist_from_nuc = ndi.distance_transform_edt(nuclei_labels == 0)
    topography = (dist_from_mito * 10.0) + dist_from_nuc

    if boundary_smoothing > 0:
        flood_mask = morphology.binary_opening(
            morphology.binary_closing(flood_mask, morphology.disk(boundary_smoothing)), 
            morphology.disk(boundary_smoothing)
        )
        topography = filters.gaussian(topography, sigma=boundary_smoothing)
    
    cell_labels = segmentation.watershed(topography, nuclei_labels, mask=flood_mask)
    
    if save_qc and out_folder is not None:
        bound_img = segmentation.mark_boundaries(img_hoechst, cell_labels, color=(1, 0, 0), mode='thin')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_hoechst, cmap='gray'); axes[0].set_title('original hoechst'); axes[0].axis('off')
        axes[1].imshow(bound_img); axes[1].set_title(f'organic cell zones ({np.max(nuclei_labels)})'); axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f"{base_name}_cell_zones_QC.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    return cell_labels

def preprocess_image(img, tophat_radius=5):
    """cleans up noise and equalizes lighting before we calculate morphology"""
    img_denoised = filters.median(img, morphology.disk(1))
    img_clahe = exposure.equalize_adapthist(img_denoised, kernel_size=24, clip_limit=0.015)
    
    selem = morphology.disk(tophat_radius)
    img_tophat = morphology.white_tophat(img_clahe, footprint=selem)
    
    return img_clahe, img_tophat

def get_mask(img_tophat, sigma=2.0, method='triangle'):
    """standard fallback method using traditional adaptive thresholding"""
    img_smoothed = filters.gaussian(img_tophat, sigma=sigma)
    
    if method == 'triangle':
        thresh = filters.threshold_triangle(img_smoothed)
    elif method == 'li':
        thresh = filters.threshold_li(img_smoothed + 1e-8)
    else:
        thresh = filters.threshold_otsu(img_smoothed)
        
    mask = img_smoothed > thresh
    mask = morphology.remove_small_objects(mask, max_size=10)
    return mask

def generate_cnn_mask_sequential(img, model, patch_h, patch_w, suppress_output=True):
    """applies the cnn using rational block-tiling"""
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img_8bit = (img_norm * 255).astype(np.float32)

    h, w = img_8bit.shape
    pad_h = (patch_h - h % patch_h) % patch_h
    pad_w = (patch_w - w % patch_w) % patch_w
    
    img_padded = np.pad(img_8bit, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = img_padded.shape

    pred_padded = np.zeros((padded_h, padded_w), dtype=np.float32)

    total_tiles = (padded_h // patch_h) * (padded_w // patch_w)
    tile_idx = 1

    for i in range(0, padded_h, patch_h):
        for j in range(0, padded_w, patch_w):
            tile = img_padded[i:i+patch_h, j:j+patch_w]
            tile_batch = np.expand_dims(np.expand_dims(tile, axis=0), axis=-1)
            
            if not suppress_output:
                print(f"\n    -> inferring tile {tile_idx}/{total_tiles}...", end="", flush=True)
            
            pred = model.predict_on_batch(tile_batch)
            
            if not suppress_output:
                print(" done.", flush=True)
            
            pred_array = pred.numpy() if hasattr(pred, 'numpy') else np.array(pred)
            pred_padded[i:i+patch_h, j:j+patch_w] = pred_array[0, :, :, 0]
            tile_idx += 1

    pred_mask = pred_padded > 0.5 
    mask = pred_mask[:h, :w]
    mask = morphology.remove_small_objects(mask, max_size=10)
    
    return mask

# ==========================================
# QUALITY CONTROL AND METRICS
# ==========================================

def save_mask_visualisation(img, mask, out_folder, base_name, overlay_color='purple', alpha=0.4, suffix="mask_vis"):
    """puts the original image and a transparent mask overlay side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title('original image'); axes[0].axis('off')
    axes[1].imshow(img, cmap='gray')
    
    rgba = np.zeros((*img.shape, 4))
    rgb_color = mcolors.to_rgb(overlay_color)
    rgba[mask] = [*rgb_color, alpha]
    
    axes[1].imshow(rgba); axes[1].set_title('mask overlay'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_tmrm_mtg_overlay_qc(img_tmrm, mask_mtg, mask_tmrm, out_folder, base_name):
    """creates a true rgb overlay to visualise physical overlap between networks and potential"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_tmrm, cmap='gray'); axes[0].set_title('original tmrm'); axes[0].axis('off')
    
    rgb_overlay = np.zeros((*img_tmrm.shape, 3))
    
    # normalize background to act as structural guide
    img_norm = (img_tmrm - np.min(img_tmrm)) / (np.max(img_tmrm) - np.min(img_tmrm) + 1e-8)
    rgb_overlay[..., 0] = img_norm * 0.3
    rgb_overlay[..., 1] = img_norm * 0.3
    rgb_overlay[..., 2] = img_norm * 0.3
    
    # green mtg, red tmrm, yellow overlap
    rgb_overlay[mask_mtg] = [0, 0.8, 0] 
    rgb_overlay[mask_tmrm] = [0.8, 0, 0] 
    rgb_overlay[mask_mtg & mask_tmrm] = [0.9, 0.9, 0] 
    
    axes[1].imshow(rgb_overlay); axes[1].set_title('green=mtg, red=tmrm, yellow=overlap'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_tmrm_mtg_overlap_QC.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_master_overlay_qc(img_base, mask_hoechst, mask_mtg, mask_tmrm, cell_labels, out_folder, base_name):
    """combines all masks and boundaries into a single rational composite qc"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_base, cmap='gray')
    
    if mask_hoechst is not None:
        blue_rgba = np.zeros((*img_base.shape, 4)); blue_rgba[mask_hoechst] = [*mcolors.to_rgb('blue'), 0.4]
        ax.imshow(blue_rgba)
    if mask_mtg is not None:
        green_rgba = np.zeros((*img_base.shape, 4)); green_rgba[mask_mtg] = [*mcolors.to_rgb('green'), 0.4]
        ax.imshow(green_rgba)
    if mask_tmrm is not None and np.any(mask_tmrm):
        red_rgba = np.zeros((*img_base.shape, 4)); red_rgba[mask_tmrm] = [*mcolors.to_rgb('red'), 0.4]
        ax.imshow(red_rgba)
    if cell_labels is not None:
        boundaries = segmentation.find_boundaries(cell_labels, mode='inner')
        bound_rgba = np.zeros((*img_base.shape, 4)); bound_rgba[boundaries] = [*mcolors.to_rgb('red'), 1.0] 
        ax.imshow(bound_rgba)
        
    ax.axis('off')
    ax.set_title('Composite: Blue=Nuclei, Green=MTG, Red_trans=TMRM, Red_solid=Cell Bounds')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_master_composite_QC.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def calculate_morphology(img_base, mask, sigma=0.8, bin_thresholds=(0.4, 0.5)):
    """calculates tubularity from hessian eigenvalues based on the constrained mask"""
    H_elems = feature.hessian_matrix(img_base, sigma=sigma, order='rc', use_gaussian_derivatives=True)
    lambda1, lambda2 = feature.hessian_matrix_eigvals(H_elems)
    
    epsilon = 1e-8
    morphology_index = np.abs(lambda1) / (np.abs(lambda2) + epsilon)
    morphology_index[lambda2 > 0] = 0
    morphology_index = np.clip(morphology_index, 0, 1)

    mask_float = mask.astype(float)
    smoothed_index_raw = filters.gaussian(morphology_index * mask_float, sigma=1.0)
    smoothed_weights = filters.gaussian(mask_float, sigma=1.0)
    
    smoothed_index = np.zeros_like(morphology_index)
    valid_pixels = smoothed_weights > 1e-5
    smoothed_index[valid_pixels] = smoothed_index_raw[valid_pixels] / smoothed_weights[valid_pixels]
    
    smoothed_index = np.clip(smoothed_index, 0, 1)
    smoothed_index[~mask] = 0

    t1, t2 = bin_thresholds
    binned_map = np.zeros_like(smoothed_index, dtype=np.int8)
    binned_map[mask & (smoothed_index < t1)] = 1
    binned_map[mask & (smoothed_index >= t1) & (smoothed_index < t2)] = 2
    binned_map[mask & (smoothed_index >= t2)] = 3
    
    return smoothed_index, binned_map

def save_morphology_qc(img_clahe, mask, smoothed_index, binned_map, bin_thresholds, out_folder, base_name):
    """dumps the continuous and discrete morphology maps for manual checking"""
    cmap_cont = LinearSegmentedColormap.from_list('tube_to_spot', ['blue', 'red'])
    colored_index = cmap_cont(smoothed_index)
    colored_index[~mask] = [0, 0, 0, 1]

    overlay_cont = np.copy(colored_index)
    overlay_cont[..., 3] = np.where(mask, 0.6, 0.0)

    fig_cont, axes_cont = plt.subplots(1, 2, figsize=(12, 6))
    axes_cont[0].imshow(img_clahe, cmap='gray'); axes_cont[0].axis('off')
    axes_cont[1].imshow(img_clahe, cmap='gray'); axes_cont[1].imshow(overlay_cont); axes_cont[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_comp_continuous.png"), dpi=300)
    plt.close(fig_cont)

    t1, t2 = bin_thresholds
    binned_rgba = np.zeros((*img_clahe.shape, 4))
    binned_rgba[binned_map == 1] = [*mcolors.to_rgb('#0072B2'), 0.6]
    binned_rgba[binned_map == 2] = [*mcolors.to_rgb('#F0E442'), 0.6]
    binned_rgba[binned_map == 3] = [*mcolors.to_rgb('#D55E00'), 0.6]

    fig_disc, axes_disc = plt.subplots(1, 2, figsize=(12, 6))
    axes_disc[0].imshow(img_clahe, cmap='gray'); axes_disc[0].axis('off')
    axes_disc[1].imshow(img_clahe, cmap='gray'); axes_disc[1].imshow(binned_rgba); axes_disc[1].axis('off')
    
    legend_elements = [
        mpatches.Patch(color='#0072B2', label=f'Tubular (< {t1})'),
        mpatches.Patch(color='#F0E442', label='Intermediate'),
        mpatches.Patch(color='#D55E00', label=f'Spotty (>= {t2})')
    ]
    axes_disc[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_comp_discrete.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_disc)


# ==========================================
# PARALLEL WORKER AND AGGREGATION
# ==========================================

def _init_worker():
    """strictly prevents workers from inheriting any buggy TF/oneDNN state on Windows"""
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'

def _worker_process_field(args):
    """isolated worker for heavy numpy/scipy operations."""
    (file_path, well_id, field_id, output_paths_dict, target_channel, 
     tmrm_channel, nuclei_channel, sigma, tophat_radius, bin_thresholds, 
     use_watershed, cyto_expansion, mito_buffer, min_cyto, boundary_smoothing, mask_method, cnn_noise_filter_radius,
     use_cnn, precomputed_mask,
     save_qc_nuclei, save_qc_mtg, save_qc_tmrm, save_qc_composite, save_qc_cnn) = args
    
    well_out_dir = output_paths_dict.get(well_id)
    if not well_out_dir: return None
        
    base_filename = os.path.basename(file_path).split('.')[0]
    hoechst_path = file_path.replace(f"d{target_channel}", f"d{nuclei_channel}")
    tmrm_path = file_path.replace(f"d{target_channel}", f"d{tmrm_channel}")

    # 1. nuclei detection (d0)
    img_hoechst = load_as_float(hoechst_path)
    base_filename_hoechst = base_filename.replace(f"d{target_channel}", f"d{nuclei_channel}")
    nuclei_labels, cell_count = segment_nuclei(img_hoechst, min_nucleus_area=500, use_watershed=use_watershed)
    mask_hoechst = nuclei_labels > 0 if nuclei_labels is not None else None

    # 2. MTG mask generation (d1)
    img_mtg = load_as_float(file_path)
    if img_mtg is None or np.max(img_mtg) == 0:
        return None
        
    img_clahe, img_tophat = preprocess_image(img_mtg, tophat_radius)
    
    if use_cnn and precomputed_mask is not None:
        mask_mtg = precomputed_mask
        
        # creates a bounding region using traditional methods to filter out cnn background noise
        if cnn_noise_filter_radius > 0:
            traditional_mask = get_mask(img_tophat, sigma=2.0, method=mask_method)
            selem = morphology.disk(cnn_noise_filter_radius)
            valid_regions = morphology.binary_dilation(traditional_mask, footprint=selem)
            mask_mtg = mask_mtg & valid_regions

        if save_qc_cnn:
            qc_cnn_dir = os.path.join(well_out_dir, "CNN Raw Mask")
            os.makedirs(qc_cnn_dir, exist_ok=True)
            save_mask_visualisation(img_clahe, mask_mtg, qc_cnn_dir, base_filename, overlay_color='green', suffix="cnn_mask_QC")
    else:
        mask_mtg = get_mask(img_tophat, sigma=2.0, method=mask_method)

    # 3. MTG morphology calculation
    smoothed_index, binned_map = calculate_morphology(img_clahe, mask_mtg, sigma=sigma, bin_thresholds=bin_thresholds)
    
    if save_qc_mtg:
        qc_mtg_dir = os.path.join(well_out_dir, "MTG Morphology")
        os.makedirs(qc_mtg_dir, exist_ok=True)
        save_morphology_qc(img_clahe, mask_mtg, smoothed_index, binned_map, bin_thresholds, qc_mtg_dir, base_filename)

    # 4. TMRM overlap (d2)
    img_tmrm = load_as_float(tmrm_path)
    mask_tmrm = np.zeros_like(mask_mtg)
    
    if img_tmrm is not None and np.max(img_tmrm) > 0:
        _, tmrm_tophat = preprocess_image(img_tmrm, tophat_radius)
        mask_tmrm = get_mask(tmrm_tophat, sigma=2.0, method=mask_method)
        
        if save_qc_tmrm:
            qc_tmrm_dir = os.path.join(well_out_dir, "TMRM Potential")
            os.makedirs(qc_tmrm_dir, exist_ok=True)
            # now overlays mtg and tmrm simultaneously 
            save_tmrm_mtg_overlay_qc(img_tmrm, mask_mtg, mask_tmrm, qc_tmrm_dir, base_filename)

    # 5. Smart cell territories
    qc_nuclei_dir = os.path.join(well_out_dir, "Hoechst Cell Zones") if save_qc_nuclei else None
    if save_qc_nuclei: os.makedirs(qc_nuclei_dir, exist_ok=True)
        
    cell_labels = define_cell_territories(
        img_hoechst, nuclei_labels, mask_mtg, mask_tmrm, 
        qc_nuclei_dir, base_filename_hoechst, 
        cyto_expansion=cyto_expansion, mito_buffer=mito_buffer, min_cyto=min_cyto, boundary_smoothing=boundary_smoothing,
        save_qc=save_qc_nuclei
    )

    # 6. Composites
    if save_qc_composite and mask_hoechst is not None and mask_mtg is not None:
        qc_comp_dir = os.path.join(well_out_dir, "Master Composite")
        os.makedirs(qc_comp_dir, exist_ok=True)
        save_master_overlay_qc(img_clahe, mask_hoechst, mask_mtg, mask_tmrm, cell_labels, qc_comp_dir, base_filename)

    # 7. Metrics extraction
    cell_records = []
    if cell_count > 0 and cell_labels is not None:
        cell_indices = np.arange(1, cell_count + 1)
        mtg_per_cell = ndi.sum(mask_mtg, labels=cell_labels, index=cell_indices)
        valid_cells = mtg_per_cell > 0 
        
        if np.any(valid_cells):
            mtg_per_cell = mtg_per_cell[valid_cells]
            valid_indices = cell_indices[valid_cells]
            
            tub_per_cell = ndi.sum(binned_map == 1, labels=cell_labels, index=valid_indices)
            int_per_cell = ndi.sum(binned_map == 2, labels=cell_labels, index=valid_indices)
            spo_per_cell = ndi.sum(binned_map == 3, labels=cell_labels, index=valid_indices)
            tmrm_per_cell = ndi.sum(mask_tmrm, labels=cell_labels, index=valid_indices)
            overlap_px_per_cell = ndi.sum(mask_mtg & mask_tmrm, labels=cell_labels, index=valid_indices)
            
            # cell area calculation across the organic territory mapped
            cell_area_per_cell = ndi.sum(np.ones_like(cell_labels), labels=cell_labels, index=valid_indices)
            tmrm_ratio_per_cell = overlap_px_per_cell / mtg_per_cell
            
            for i in range(len(valid_indices)):
                # generates a strictly unique identifier for global traceability
                global_id = f"{well_id}_f{field_id}_c{valid_indices[i]}"
                
                cell_records.append({
                    "Global_Cell_ID": global_id,
                    "Well_ID": well_id, 
                    "Field": field_id, 
                    "Local_Cell_ID": valid_indices[i],
                    "Cell_Area": cell_area_per_cell[i],
                    "Mito_Px": mtg_per_cell[i],
                    "Tubular_Px": tub_per_cell[i],
                    "Interme_Px": int_per_cell[i],
                    "Spotty_Px": spo_per_cell[i],
                    "TMRM_Px": tmrm_per_cell[i],
                    "TMRM_Overlap_Ratio": tmrm_ratio_per_cell[i]
                })

    return cell_records

def add_stat_annotation(ax, data, x_col, y_col, ref_cond):
    """calculates independent t-tests against a reference condition and draws brackets"""
    y_max = data[y_col].max()
    y_range = data[y_col].max() - data[y_col].min()
    if pd.isna(y_range) or y_range == 0: y_range = 1
    
    ref_data = data[data[x_col] == ref_cond][y_col].dropna()
    if len(ref_data) < 2: return 
    
    x_ticks = [t.get_text() for t in ax.get_xticklabels()]
    if ref_cond not in x_ticks: return
    ref_idx = x_ticks.index(ref_cond)
    
    step = 0.08 * y_range
    current_y = y_max + step
    
    for i, cond in enumerate(x_ticks):
        if cond == ref_cond: continue
        comp_data = data[data[x_col] == cond][y_col].dropna()
        if len(comp_data) < 2: continue
        
        stat, p = stats.ttest_ind(ref_data, comp_data, equal_var=False)
        
        if p < 0.05:
            stars = '*' if p >= 0.01 else '**' if p >= 0.001 else '***'
            # draws significance bracket
            ax.plot([ref_idx, ref_idx, i, i], [current_y, current_y+step/3, current_y+step/3, current_y], lw=1.5, c='k')
            ax.text((ref_idx + i) / 2, current_y + step/3, stars, ha='center', va='bottom', color='k')
            current_y += step * 1.5

def aggregate_and_plot_data(all_cell_data, output_paths_dict, treatment_map, csv_path_override=None):
    """saves a massive single-cell dataframe or loads an existing one, then handles statistical boxplots"""
    if csv_path_override and os.path.exists(csv_path_override):
        df_cells = pd.read_csv(csv_path_override)
        parent_out_dir = os.path.dirname(csv_path_override)
        print(f"\nloaded existing single-cell dataset from: {csv_path_override}")
    else:
        if not all_cell_data: return
        
        parent_out_dir = os.path.dirname(output_paths_dict[all_cell_data[0]["Well_ID"]])
        csv_path = os.path.join(parent_out_dir, "Plate_SingleCell_Morphology_Data.csv")
        
        # master pandas dataframe creation
        df_cells = pd.DataFrame(all_cell_data)
        
        # maps condition and rationally reorders columns for better readability
        df_cells['Condition'] = df_cells['Well_ID'].map(treatment_map).fillna("Untreated")
        cols = ['Global_Cell_ID', 'Well_ID', 'Condition', 'Field', 'Local_Cell_ID'] + [c for c in df_cells.columns if c not in ['Global_Cell_ID', 'Well_ID', 'Condition', 'Field', 'Local_Cell_ID']]
        df_cells = df_cells[cols]
        
        # save the heavy single-cell dataframe so we never have to re-extract for custom plotting
        df_cells.to_csv(csv_path, index=False)
        print(f"\nexported single-cell master dataset to: {csv_path}")
            
    # aggregate to well level (median) to strictly avoid pseudoreplication in statistics
    well_df = df_cells.groupby(['Well_ID', 'Condition']).agg(
        Cell_Count=('Global_Cell_ID', 'count'),
        Cell_Area=('Cell_Area', 'median'),
        Tubular=('Tubular_Px', 'median'),
        Intermediate=('Interme_Px', 'median'),
        Spotty=('Spotty_Px', 'median'),
        TMRM_Ratio=('TMRM_Overlap_Ratio', 'median')
    ).reset_index()
    
    conditions = sorted(well_df['Condition'].unique())
    ref_cond = "Vehicle" if "Vehicle" in conditions else conditions[0]

    sns.set_style("whitegrid")
    
    metrics = [
        ('Cell_Count', 'Total Cells per Well', 'grey'),
        ('Cell_Area', 'Cell Area (Px)', 'brown'),
        ('Tubular', 'Tubular Network (Px/Cell)', '#0072B2'),
        ('Intermediate', 'Intermediate Network (Px/Cell)', '#F0E442'),
        ('Spotty', 'Fragmented Network (Px/Cell)', '#D55E00'),
        ('TMRM_Ratio', 'Membrane Potential (TMRM/MTG Ratio)', 'purple')
    ]
    
    # setup text file for logging outliers rationally
    outlier_log_path = os.path.join(parent_out_dir, "Analysis outliers.txt")
    with open(outlier_log_path, "w") as outlier_log:
        outlier_log.write("Identification of statistical outliers (1.5 * iqr)\n")
        outlier_log.write("="*60 + "\n")
        
        # generates and saves a separate, cleanly proportioned plot for each metric
        for col, title, color in metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            outlier_log.write(f"\n--- {title} ({col}) ---\n")
            outliers_found_for_metric = False
            
            # carefully partition the data so stripplot doesn't paint filled circles over our hollow outliers
            non_outlier_indices = []
            for cond in conditions:
                cond_data = well_df[well_df['Condition'] == cond]
                if len(cond_data) < 3:
                    non_outlier_indices.extend(cond_data.index)
                    continue
                
                # exactly matches matplotlib's native boxplot stats
                q1, q3 = np.percentile(cond_data[col].dropna(), [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = cond_data[(cond_data[col] < lower_bound) | (cond_data[col] > upper_bound)]
                normals = cond_data[(cond_data[col] >= lower_bound) & (cond_data[col] <= upper_bound)]
                
                if not outliers.empty:
                    outliers_found_for_metric = True
                    outlier_log.write(f"  > {cond}:\n")
                    for _, row in outliers.iterrows():
                        outlier_log.write(f"      well {row['Well_ID']} - value: {row[col]:.3f}\n")
                        
                non_outlier_indices.extend(normals.index)
                
            if not outliers_found_for_metric:
                outlier_log.write("  > no outliers detected.\n")
                
            df_normals = well_df.loc[non_outlier_indices]
            
            # standard boxplot with strictly unfilled outlier markers
            flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=7, alpha=0.8)
            sns.boxplot(data=well_df, x='Condition', y=col, ax=ax, color=color, showfliers=True, flierprops=flierprops, width=0.5, boxprops=dict(alpha=0.6))
            
            # plot the internal points ONLY for the normal distribution range
            if not df_normals.empty:
                sns.stripplot(data=df_normals, x='Condition', y=col, ax=ax, color='black', alpha=0.7, jitter=True, size=6)
            
            add_stat_annotation(ax, well_df, 'Condition', col, ref_cond)
            
            ax.set_ylabel(title)
            ax.set_xlabel('')
            ax.set_title(title.split('(')[0].strip(), fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # fixes the broken x-axis labels using intelligent text wrapping
            labels = [item.get_text() for item in ax.get_xticklabels()]
            wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in labels]
            ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=9)
            
            plt.tight_layout()
            plot_filename = f"Condition_{col}_Boxplot.png"
            plt.savefig(os.path.join(parent_out_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close(fig)

def process_plate_directory(
    data_dir, output_paths_dict, treatment_map, 
    target_channel="1", tmrm_channel="2", nuclei_channel="0", 
    sigma=1.5, tophat_radius=5, test_well=None, bin_thresholds=(0.4, 0.5), 
    use_watershed=False, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3,
    mask_method='triangle', cnn_noise_filter_radius=25,
    use_cnn=False, cnn_model_path=None,
    save_qc_nuclei=True, save_qc_mtg=True, save_qc_tmrm=True, save_qc_composite=True, save_qc_cnn=True,
    plot_only=False, master_csv_path=None
):
    # rationally short-circuits the entire image processing pipeline if we just want to re-plot
    if plot_only:
        print("\n--- phase: plotting only (skipping image extraction) ---")
        if master_csv_path and os.path.exists(master_csv_path):
            aggregate_and_plot_data(None, None, None, csv_path_override=master_csv_path)
        else:
            print(f"error: plot_only is true but valid master_csv_path was not found at {master_csv_path}")
        return

    search_pattern = os.path.join(data_dir, "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern)
    pattern = re.compile(r'_([A-H]\d{2})f(\d{2})d(\d)')
    
    target_files = [f for f in all_files if pattern.search(f) and pattern.search(f).group(3) == target_channel]
    if test_well:
        target_files = [f for f in target_files if pattern.search(f).group(1) == test_well]
        
    cnn_masks_dict = {}
    
    if use_cnn and cnn_model_path:
        print("\n--- Phase 1: Sequential CNN Inference ---")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['OMP_NUM_THREADS'] = '1' 
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
        
        import tensorflow as tf
        import tensorflow.keras.backend as K
        
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(cnn_model_path, compile=False)
        patch_h, patch_w = model.input_shape[1:3]
        if patch_h is None: patch_h, patch_w = 656, 656
            
        print("Model loaded. Generating structural masks...\n")
        
        for f in tqdm(target_files, desc="CNN Prediction"):
            img_mtg = load_as_float(f)
            if img_mtg is None or np.max(img_mtg) == 0: continue
            img_clahe, _ = preprocess_image(img_mtg, tophat_radius)
            cnn_masks_dict[f] = generate_cnn_mask_sequential(img_clahe, model, patch_h, patch_w, suppress_output=True)

        print("--- Phase 1 Complete. Clearing Neural Engine... ---")
        # radical memory and thread lock purging
        del model
        K.clear_session()
        gc.collect()
        print("Neural engine purged.\n")

    print("\n--- Phase 2: Parallel Morphology & Extraction ---")
    tasks = []
    for f in target_files:
        match = pattern.search(f)
        tasks.append((
            f, match.group(1), match.group(2), output_paths_dict, target_channel, tmrm_channel, nuclei_channel, 
            sigma, tophat_radius, bin_thresholds, use_watershed, 
            cyto_expansion, mito_buffer, min_cyto, boundary_smoothing, mask_method, cnn_noise_filter_radius,
            use_cnn, cnn_masks_dict.get(f), 
            save_qc_nuclei, save_qc_mtg, save_qc_tmrm, save_qc_composite, save_qc_cnn
        ))

    all_cell_data = []
    # on windows with cnn masks in ram, we limit to 6 cores max to prevent pickle-overhead lag
    max_workers = min(6, os.cpu_count() - 1, len(tasks)) if os.cpu_count() else 1
    
    print(f"Spawning {max_workers} isolated processes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_worker_process_field, t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Calculating Morphology"):
            res = future.result()
            if res: all_cell_data.extend(res)

    if all_cell_data:
        aggregate_and_plot_data(all_cell_data, output_paths_dict, treatment_map)


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    base_dir = r"C:/Users/Thomas/Desktop/Temp local/General NQO2 Search/Morphology"
    data_path= r"C:/Users/Thomas/Vandria/Superlab - 4 NQO2 Biology/Mitophagy/Microscopy/04 - TMRM & MitoTracker/TCH_003_TMRM_MTG_HepG2/Plate 3/AcquireOnly.V3_04-09-26_05;16;16"
    dtt_path = "C:/Users/Thomas/Vandria/Superlab - 4 NQO2 Biology/Mitophagy/Microscopy/04 - TMRM & MitoTracker/TCH_003_TMRM_MTG_HepG2/Plate 3/Drug treatment plate 3.tdd"
    cnn_model_file = r"C:/Users/Thomas/Desktop/Temp local/General NQO2 Search/Morphology/MitoSegNet_model.hdf5"
    
    out_dir, paths_dict = setup_output(base_dir, "Output Images plate 3", zero_pad=True)
    treatment_dictionary = parse_tecan_d300e_xml(dtt_path)
    
    # define the path to your previously generated master csv
    existing_csv_path = os.path.join(out_dir, "Plate_SingleCell_Morphology_Data.csv")
    
    process_plate_directory(
        data_dir=data_path, output_paths_dict=paths_dict, treatment_map=treatment_dictionary,
        target_channel="1", tmrm_channel="2", nuclei_channel="0", 
        use_cnn=True, cnn_model_path=cnn_model_file,
        sigma=1.5, tophat_radius=6, test_well=None, bin_thresholds=(0.4, 0.5),
        use_watershed=True, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3,
        mask_method='triangle', cnn_noise_filter_radius=10,
        save_qc_nuclei=True, save_qc_mtg=True, save_qc_tmrm=True, save_qc_composite=True, save_qc_cnn=True,
        plot_only=True, master_csv_path=existing_csv_path
    )