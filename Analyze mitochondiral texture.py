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
from skimage import io, filters, feature, exposure, morphology, color, segmentation, measure
from matplotlib.colors import LinearSegmentedColormap
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

def define_cell_territories(img_hoechst, nuclei_labels, mask_mtg, out_folder, base_name, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3, save_qc=True):
    """uses a topographical watershed to organically wrap around the mitochondrial networks."""
    if nuclei_labels is None:
        return None
        
    mito_footprint = np.zeros(nuclei_labels.shape, dtype=bool)
    if mask_mtg is not None:
        mito_footprint |= mask_mtg
        
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
        fig.clf()
        plt.close('all')
        
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

def generate_cnn_mask_sequential(img, sess, input_name, patch_h, patch_w, suppress_output=False):
    """applies the cnn using ONNX runtime and rational block-tiling"""
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
            # ONNX expects float32 arrays
            tile_batch = np.expand_dims(np.expand_dims(tile, axis=0), axis=-1).astype(np.float32)
            
            if not suppress_output:
                print(f"      [Tile {tile_idx:02d}/{total_tiles:02d}] inferring on GPU...", end="\r", flush=True)
            
            # ONNX execution
            pred = sess.run(None, {input_name: tile_batch})[0]
            
            pred_padded[i:i+patch_h, j:j+patch_w] = pred[0, :, :, 0]
            tile_idx += 1
            
    if not suppress_output:
        print(" " * 65, end="\r") # clear the line once done

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
    
    # memory fix: force uint8 (8-bit) array instead of float64 (64-bit) to prevent RAM explosion
    rgba = np.zeros((*img.shape, 4), dtype=np.uint8)
    rgb_color = mcolors.to_rgb(overlay_color)
    rgba[mask] = [int(c * 255) for c in rgb_color] + [int(alpha * 255)]
    
    axes[1].imshow(rgba); axes[1].set_title('mask overlay'); axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_{suffix}.png"), dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close('all')

def save_tmrm_mtg_overlay_qc(img_tmrm, mask_mtg, out_folder, base_name):
    """visualizes physical overlap between TMRM intensities and MTG boundaries"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_tmrm, cmap='magma'); axes[0].set_title('Raw TMRM Intensities'); axes[0].axis('off')
    
    axes[1].imshow(img_tmrm, cmap='magma')
    axes[1].contour(mask_mtg, colors='green', linewidths=0.7)
    axes[1].set_title('TMRM mapping within MTG network'); axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_tmrm_mapping_QC.png"), dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close('all')

def save_master_overlay_qc(img_base, mask_hoechst, mask_mtg, cell_labels, out_folder, base_name):
    """combines all structural boundaries into a single rational composite qc"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_base, cmap='gray')
    
    # memory fix: force uint8 arrays. float64 arrays of 2843x2843x4 take exactly 247 MiB of RAM each!
    if mask_hoechst is not None:
        blue_rgba = np.zeros((*img_base.shape, 4), dtype=np.uint8)
        blue_rgba[mask_hoechst] = [0, 0, 255, int(0.4 * 255)]
        ax.imshow(blue_rgba)
    if mask_mtg is not None:
        green_rgba = np.zeros((*img_base.shape, 4), dtype=np.uint8)
        green_rgba[mask_mtg] = [0, 128, 0, int(0.4 * 255)]
        ax.imshow(green_rgba)
    if cell_labels is not None:
        boundaries = segmentation.find_boundaries(cell_labels, mode='inner')
        bound_rgba = np.zeros((*img_base.shape, 4), dtype=np.uint8)
        bound_rgba[boundaries] = [255, 0, 0, 255] 
        ax.imshow(bound_rgba)
        
    ax.axis('off')
    ax.set_title('Composite: Blue=Nuclei, Green=MTG, Red=Cell Bounds')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_master_composite_QC.png"), dpi=300, bbox_inches='tight')
    fig.clf()
    plt.close('all')

def calculate_morphology(img_base, mask, sigma=0.8):
    """calculates continuous tubularity index from hessian eigenvalues (0.0 to 1.0)"""
    H_elems = feature.hessian_matrix(img_base, sigma=sigma, order='rc', use_gaussian_derivatives=True)
    lambda1, lambda2 = feature.hessian_matrix_eigvals(H_elems)
    
    epsilon = 1e-8
    # lambda2 is the cross-sectional curvature (large absolute value for tubes)
    # lambda1 is the longitudinal curvature (small absolute value for tubes)
    # the ratio approaches 0 for tubular structures and 1 for spotty/circular structures
    ratio = np.abs(lambda1) / (np.abs(lambda2) + epsilon)
    
    # invert the scale so 1.0 = highly tubular and 0.0 = spotty
    morphology_index = 1.0 - ratio
    
    # dark background means eigenvalues should be negative for bright ridges
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
    
    return smoothed_index

def save_morphology_qc(img_clahe, mask, smoothed_index, out_folder, base_name):
    """dumps the continuous morphology map for manual checking"""
    cmap_cont = LinearSegmentedColormap.from_list('spot_to_tube', ['red', 'blue'])
    
    # memory fix: cmap returns float64, immediately scale and cast to uint8
    colored_index = (cmap_cont(smoothed_index) * 255).astype(np.uint8)
    colored_index[~mask] = [0, 0, 0, 255]

    overlay_cont = np.copy(colored_index)
    overlay_cont[..., 3] = np.where(mask, int(0.6 * 255), 0)

    fig_cont, axes_cont = plt.subplots(1, 2, figsize=(12, 6))
    axes_cont[0].imshow(img_clahe, cmap='gray'); axes_cont[0].axis('off')
    axes_cont[1].imshow(img_clahe, cmap='gray'); axes_cont[1].imshow(overlay_cont); axes_cont[1].axis('off')
    axes_cont[1].set_title("Tubularness: Red=0 (Spotty), Blue=1 (Tubular)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_comp_continuous.png"), dpi=300)
    fig_cont.clf()
    plt.close('all')


# ==========================================
# PARALLEL WORKER AND AGGREGATION
# ==========================================

def _worker_process_field(args):
    """isolated worker for heavy numpy/scipy extraction operations."""
    (file_path, well_id, field_id, output_paths_dict, target_channel, 
     tmrm_channel, nuclei_channel, sigma, tophat_radius, 
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
    smoothed_index = calculate_morphology(img_clahe, mask_mtg, sigma=sigma)
    
    if save_qc_mtg:
        qc_mtg_dir = os.path.join(well_out_dir, "MTG Morphology")
        os.makedirs(qc_mtg_dir, exist_ok=True)
        save_morphology_qc(img_clahe, mask_mtg, smoothed_index, qc_mtg_dir, base_filename)

    # 4. TMRM alignment (d2)
    img_tmrm = load_as_float(tmrm_path)
    
    if save_qc_tmrm and img_tmrm is not None:
        qc_tmrm_dir = os.path.join(well_out_dir, "TMRM Potential")
        os.makedirs(qc_tmrm_dir, exist_ok=True)
        save_tmrm_mtg_overlay_qc(img_tmrm, mask_mtg, qc_tmrm_dir, base_filename)

    # 5. Smart cell territories
    qc_nuclei_dir = os.path.join(well_out_dir, "Hoechst Cell Zones") if save_qc_nuclei else None
    if save_qc_nuclei: os.makedirs(qc_nuclei_dir, exist_ok=True)
        
    cell_labels = define_cell_territories(
        img_hoechst, nuclei_labels, mask_mtg,
        qc_nuclei_dir, base_filename_hoechst, 
        cyto_expansion=cyto_expansion, mito_buffer=mito_buffer, min_cyto=min_cyto, boundary_smoothing=boundary_smoothing,
        save_qc=save_qc_nuclei
    )

    # 6. Composites
    if save_qc_composite and mask_hoechst is not None and mask_mtg is not None:
        qc_comp_dir = os.path.join(well_out_dir, "Master Composite")
        os.makedirs(qc_comp_dir, exist_ok=True)
        save_master_overlay_qc(img_clahe, mask_hoechst, mask_mtg, cell_labels, qc_comp_dir, base_filename)

    # 7. Single Cell Extraction (Histograms & Means)
    cell_records = []
    if cell_count > 0 and cell_labels is not None:
        props_cells = measure.regionprops(cell_labels)
        area_dict = {p.label: p.area for p in props_cells}
        
        # Mask out non-mito pixels so regionprops only looks at structural networks
        mtg_labels = cell_labels * mask_mtg
        props_morph = measure.regionprops(mtg_labels, intensity_image=smoothed_index)
        
        tmrm_dict = {}
        if img_tmrm is not None:
            props_tmrm = measure.regionprops(mtg_labels, intensity_image=img_tmrm)
            tmrm_dict = {p.label: p.intensity_image[p.image] for p in props_tmrm}
        
        for prop in props_morph:
            c_id = prop.label
            morph_vals = prop.intensity_image[prop.image]
            
            if len(morph_vals) == 0: continue
            
            # create 20 bins capturing 0.0 to 1.0 distribution
            hist, _ = np.histogram(morph_vals, bins=20, range=(0.0, 1.0))
            mean_morph = np.mean(morph_vals)
            mtg_px = len(morph_vals)
            
            t_vals = tmrm_dict.get(c_id, [])
            mean_tmrm = np.mean(t_vals) if len(t_vals) > 0 else 0.0
            
            global_id = f"{well_id}_f{field_id}_c{c_id}"
            
            record = {
                "Global_Cell_ID": global_id,
                "Well_ID": well_id, 
                "Field": field_id, 
                "Local_Cell_ID": c_id,
                "Cell_Area": area_dict.get(c_id, 0),
                "Mito_Px": mtg_px,
                "Mean_Morphology": mean_morph,
                "Mean_TMRM_Intensity": mean_tmrm
            }
            
            for b in range(20):
                record[f"Hist_Bin_{b+1:02d}"] = hist[b]
                
            cell_records.append(record)

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
            ax.plot([ref_idx, ref_idx, i, i], [current_y, current_y+step/3, current_y+step/3, current_y], lw=1.5, c='k')
            ax.text((ref_idx + i) / 2, current_y + step/3, stars, ha='center', va='bottom', color='k')
            current_y += step * 1.5

def aggregate_and_plot_data(all_cell_data, output_paths_dict, treatment_map, csv_path_override=None):
    """saves single-cell morphological and potential data, plots continuous distribution curves, and boxplots"""
    if csv_path_override and os.path.exists(csv_path_override):
        df_cells = pd.read_csv(csv_path_override)
        parent_out_dir = os.path.dirname(csv_path_override)
        print(f"\nLoaded existing single-cell dataset from: {csv_path_override}")
    else:
        if not all_cell_data: return
        
        parent_out_dir = os.path.dirname(output_paths_dict[all_cell_data[0]["Well_ID"]])
        csv_path = os.path.join(parent_out_dir, "Plate_SingleCell_Morphology_Data.csv")
        
        df_cells = pd.DataFrame(all_cell_data)
        df_cells['Condition'] = df_cells['Well_ID'].map(treatment_map).fillna("Untreated")
        
        meta_cols = ['Global_Cell_ID', 'Well_ID', 'Condition', 'Field', 'Local_Cell_ID', 'Cell_Area', 'Mito_Px', 'Mean_Morphology', 'Mean_TMRM_Intensity']
        hist_cols = [c for c in df_cells.columns if c.startswith('Hist_Bin_')]
        cols = meta_cols + hist_cols
        df_cells = df_cells[cols]
        
        df_cells.to_csv(csv_path, index=False)
        print(f"\nExported single-cell master dataset to: {csv_path}")
            
    agg_funcs = {
        'Global_Cell_ID': 'count',
        'Cell_Area': 'median',
        'Mean_Morphology': 'median',
        'Mean_TMRM_Intensity': 'median'
    }
    for b in range(1, 21):
        if f'Hist_Bin_{b:02d}' in df_cells.columns:
            agg_funcs[f'Hist_Bin_{b:02d}'] = 'sum'
            
    well_df = df_cells.groupby(['Well_ID', 'Condition']).agg(agg_funcs).rename(columns={'Global_Cell_ID': 'Cell_Count'}).reset_index()
    
    conditions = sorted(well_df['Condition'].unique())
    ref_cond = "Vehicle" if "Vehicle" in conditions else conditions[0]
    sns.set_style("whitegrid")
    
    # ----------------------------------------------------
    # 1. Plot Continuous Morphology Distributions
    # ----------------------------------------------------
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    bin_centers = np.linspace(0.025, 0.975, 20)
    colors = sns.color_palette("husl", len(conditions))
    
    for i, cond in enumerate(conditions):
        cond_data = well_df[well_df['Condition'] == cond]
        if cond_data.empty: continue
        
        norm_hists = []
        for _, row in cond_data.iterrows():
            if row['Cell_Count'] > 0:
                hist_vals = np.array([row.get(f'Hist_Bin_{b:02d}', 0) for b in range(1, 21)])
                norm_hists.append(hist_vals / row['Cell_Count'])
                
        if norm_hists:
            norm_hists = np.array(norm_hists)
            mean_hist = np.mean(norm_hists, axis=0)
            sem_hist = stats.sem(norm_hists, axis=0) if len(norm_hists) > 1 else np.zeros_like(mean_hist)
            
            line, = ax_dist.plot(bin_centers, mean_hist, label=cond, color=colors[i], lw=2)
            ax_dist.fill_between(bin_centers, mean_hist - sem_hist, mean_hist + sem_hist, alpha=0.2, color=colors[i])

    ax_dist.set_title("Continuous Mitochondrial Tubularness Distribution", fontsize=14, fontweight='bold')
    ax_dist.set_xlabel("Tubularness Index (0=Spotty, 1=Tubular)")
    ax_dist.set_ylabel("Average Pixel Area per Cell")
    ax_dist.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(parent_out_dir, "Condition_Morphology_Distribution.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_dist)

    # ----------------------------------------------------
    # 2. Boxplot Analytics 
    # ----------------------------------------------------
    metrics = [
        ('Cell_Count', 'Total Cells per Well', 'grey'),
        ('Cell_Area', 'Cell Area (Px)', 'brown'),
        ('Mean_Morphology', 'Mean Tubularness Index per Well', '#0072B2'),
        ('Mean_TMRM_Intensity', 'Mean TMRM Intensity per Well', 'purple')
    ]
    
    outlier_log_path = os.path.join(parent_out_dir, "Analysis_Outliers.txt")
    with open(outlier_log_path, "w") as outlier_log:
        outlier_log.write("Identification of statistical outliers (1.5 * iqr)\n")
        outlier_log.write("="*60 + "\n")
        
        for col, title, color in metrics:
            if col not in well_df.columns: continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            outlier_log.write(f"\n--- {title} ({col}) ---\n")
            outliers_found_for_metric = False
            non_outlier_indices = []
            
            for cond in conditions:
                cond_data = well_df[well_df['Condition'] == cond]
                if len(cond_data) < 3:
                    non_outlier_indices.extend(cond_data.index)
                    continue
                
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
                        outlier_log.write(f"     well {row['Well_ID']} - value: {row[col]:.3f}\n")
                        
                non_outlier_indices.extend(normals.index)
                
            if not outliers_found_for_metric:
                outlier_log.write("  > no outliers detected.\n")
                
            df_normals = well_df.loc[non_outlier_indices]
            
            flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=7, alpha=0.8)
            sns.boxplot(data=well_df, x='Condition', y=col, ax=ax, color=color, showfliers=True, flierprops=flierprops, width=0.5, boxprops=dict(alpha=0.6))
            
            if not df_normals.empty:
                sns.stripplot(data=df_normals, x='Condition', y=col, ax=ax, color='black', alpha=0.7, jitter=True, size=6)
            
            add_stat_annotation(ax, well_df, 'Condition', col, ref_cond)
            
            ax.set_ylabel(title)
            ax.set_xlabel('')
            ax.set_title(title.split('(')[0].strip(), fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            labels = [item.get_text() for item in ax.get_xticklabels()]
            wrapped_labels = ['\n'.join(textwrap.wrap(label, width=20)) for label in labels]
            ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=9)
            
            plt.tight_layout()
            plot_filename = f"Condition_{col}_Boxplot.png"
            plt.savefig(os.path.join(parent_out_dir, plot_filename), dpi=300, bbox_inches='tight')
            fig.clf()
            plt.close('all')

def process_plate_directory(
    data_dir, output_paths_dict, treatment_map, 
    target_channel="1", tmrm_channel="2", nuclei_channel="0", 
    sigma=1.5, tophat_radius=5, test_well=None,
    use_watershed=False, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3,
    mask_method='triangle', cnn_noise_filter_radius=25,
    use_cnn=False, cnn_model_path=None,
    save_qc_nuclei=True, save_qc_mtg=True, save_qc_tmrm=True, save_qc_composite=True, save_qc_cnn=True,
    plot_only=False, master_csv_path=None
):
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
        print("\n--- Phase 1: ONNX GPU Inference ---")
        try:
            import onnxruntime as ort
        except ImportError:
            print("\n[!] ERROR: onnxruntime is not installed.")
            print("Please run: pip install onnxruntime-directml\n")
            return

        # Request DirectML (Windows GPU) first, fallback to CPU if it fails
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        print("Loading ONNX Model into VRAM via DirectX 12...")
        try:
            sess = ort.InferenceSession(cnn_model_path, providers=providers)
            active_provider = sess.get_providers()[0]
            if active_provider == 'DmlExecutionProvider':
                print("[SUCCESS] RTX 3060Ti active and ready.")
            else:
                print(f"[WARNING] DirectML failed to load. Falling back to CPU ({active_provider}).")
                
            input_name = sess.get_inputs()[0].name
            input_shape = sess.get_inputs()[0].shape
            
            # Handle dynamic shapes gracefully
            patch_h = input_shape[1] if isinstance(input_shape[1], int) else 656
            patch_w = input_shape[2] if isinstance(input_shape[2], int) else 656
                
            print(f"Model loaded. Generating structural masks with patch size ({patch_h}x{patch_w})...\n")
            
            for f in tqdm(target_files, desc="Processing Images"):
                img_mtg = load_as_float(f)
                if img_mtg is None or np.max(img_mtg) == 0: continue
                img_clahe, _ = preprocess_image(img_mtg, tophat_radius)
                
                # Execute inference using ONNX
                cnn_masks_dict[f] = generate_cnn_mask_sequential(img_clahe, sess, input_name, patch_h, patch_w, suppress_output=False)

            print("\n--- Phase 1 Complete. Releasing VRAM... ---")
            del sess
            gc.collect()
            print("VRAM cleared.\n")
            
        except Exception as e:
            print(f"\n[!] ERROR running ONNX model: {e}")
            print("Ensure you have provided the correct .onnx file path.")
            return

    print("\n--- Phase 2: Parallel Morphology & Extraction ---")
    tasks = []
    for f in target_files:
        match = pattern.search(f)
        tasks.append((
            f, match.group(1), match.group(2), output_paths_dict, target_channel, tmrm_channel, nuclei_channel, 
            sigma, tophat_radius, use_watershed, 
            cyto_expansion, mito_buffer, min_cyto, boundary_smoothing, mask_method, cnn_noise_filter_radius,
            use_cnn, cnn_masks_dict.get(f), 
            save_qc_nuclei, save_qc_mtg, save_qc_tmrm, save_qc_composite, save_qc_cnn
        ))

    all_cell_data = []
    max_workers = min(6, os.cpu_count() - 1, len(tasks)) if os.cpu_count() else 1
    
    print(f"Spawning {max_workers} isolated processes...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    
    cnn_model_file = r"C:/Users/Thomas/Desktop/Temp local/General NQO2 Search/Morphology/MitoSegNet_model.onnx"
    
    out_dir, paths_dict = setup_output(base_dir, "Output Images plate 3", zero_pad=True)
    treatment_dictionary = parse_tecan_d300e_xml(dtt_path)
    
    existing_csv_path = os.path.join(out_dir, "Plate_SingleCell_Morphology_Data.csv")
    
    process_plate_directory(
        data_dir=data_path, output_paths_dict=paths_dict, treatment_map=treatment_dictionary,
        target_channel="1", tmrm_channel="2", nuclei_channel="0", 
        use_cnn=True, cnn_model_path=cnn_model_file,
        sigma=1.5, tophat_radius=6, test_well=None,
        use_watershed=True, cyto_expansion=80, mito_buffer=20, min_cyto=20, boundary_smoothing=3,
        mask_method='triangle', cnn_noise_filter_radius=10,
        save_qc_nuclei=True, save_qc_mtg=True, save_qc_tmrm=True, save_qc_composite=True, save_qc_cnn=True,
        plot_only=False, master_csv_path=existing_csv_path
    )