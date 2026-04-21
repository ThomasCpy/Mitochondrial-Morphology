import os
import re
import csv
import glob
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import concurrent.futures
from skimage import io, filters, feature, exposure, morphology, color
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm

def setup_output(base_path, folder_name="Output masks", zero_pad=True):
    """Creates output directory and subdirectories for a 96-well plate."""
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

def count_nuclei(image_path, out_folder, base_name, min_nucleus_area=500):
    """
    Reads the Hoechst (d0) channel, smooths noise, removes debris, counts valid nuclei,
    and saves a visual QC overlay to verify the segmentation accuracy.
    """
    if not os.path.exists(image_path):
        return np.nan 

    img = io.imread(image_path, as_gray=True)
    
    if np.max(img) == 0:
        return 0
        
    img_smoothed = filters.gaussian(img, sigma=2.0)
    thresh = filters.threshold_otsu(img_smoothed)
    mask = img_smoothed > thresh
    
    mask_cleaned = morphology.remove_small_objects(mask, min_size=min_nucleus_area)
    labels, num_cells = ndi.label(mask_cleaned)
    
    image_label_overlay = color.label2rgb(labels, image=img, bg_label=0, bg_color=(0, 0, 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Hoechst')
    axes[0].axis('off')
    
    axes[1].imshow(image_label_overlay)
    axes[1].set_title(f'Detected Nuclei (Count: {num_cells})')
    axes[1].axis('off')
    
    plt.tight_layout()
    qc_out_path = os.path.join(out_folder, f"{base_name}_nuclei_QC.png")
    plt.savefig(qc_out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return num_cells

def analyze_and_save(image_path, out_folder, base_name, sigma=1.5, bin_thresholds=(0.33, 0.66)):
    """
    Applies CLAHE, Otsu, and Hessian-based tubularity analysis.
    Saves side-by-side visuals for both continuous and discrete analyses (with legends).
    """
    img = io.imread(image_path, as_gray=True)
    
    if img.dtype != np.float64 and img.dtype != np.float32:
        max_val = np.max(img)
        img_float = img.astype(np.float32) / (max_val if max_val > 0 else 1)
    else:
        img_float = img

    img_clahe = exposure.equalize_adapthist(img_float, kernel_size=24 ,clip_limit=0.03)


    selem = morphology.disk(5)
    img_cleaned = morphology.white_tophat(img_clahe, footprint= selem)

    thresh = filters.threshold_otsu(img_cleaned)
    mask = img_cleaned > thresh

    H_elems = feature.hessian_matrix(img_cleaned, sigma=sigma, order='rc', use_gaussian_derivatives=True)
    lambda1, lambda2 = feature.hessian_matrix_eigvals(H_elems)
    
    epsilon = 1e-8
    morphology_index = np.abs(lambda1) / (np.abs(lambda2) + epsilon)
    morphology_index[lambda2 > 0] = 0 
    morphology_index = np.clip(morphology_index, 0, 1)

    smooth_sigma = 1.0 
    mask_float = mask.astype(float)
    
    smoothed_index_raw = filters.gaussian(morphology_index * mask_float, sigma=smooth_sigma)
    smoothed_weights = filters.gaussian(mask_float, sigma=smooth_sigma)
    
    smoothed_index = np.zeros_like(morphology_index)
    valid_pixels = smoothed_weights > 1e-5 
    smoothed_index[valid_pixels] = smoothed_index_raw[valid_pixels] / smoothed_weights[valid_pixels]
    
    smoothed_index = np.clip(smoothed_index, 0, 1)
    smoothed_index[~mask] = 0 

    # --- 1. Continuous Side-by-Side ---
    cmap_cont = LinearSegmentedColormap.from_list('tube_to_spot', ['blue', 'red'])
    colored_index = cmap_cont(smoothed_index)
    
    overlay_cont = np.copy(colored_index)
    overlay_cont[..., 3] = np.where(mask, 0.6, 0.0)

    fig_cont, axes_cont = plt.subplots(1, 2, figsize=(12, 6))
    axes_cont[0].imshow(img_clahe, cmap='gray')
    axes_cont[0].set_title('Normalized Input (CLAHE)')
    axes_cont[0].axis('off')
    
    axes_cont[1].imshow(img_clahe, cmap='gray')
    axes_cont[1].imshow(overlay_cont)
    axes_cont[1].set_title(f'Continuous Tubularity ($\sigma$: {sigma})')
    axes_cont[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_comp_continuous.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_cont)

    # --- 2. Discrete Bins Side-by-Side (With Legend) ---
    t1, t2 = bin_thresholds
    binned_map = np.zeros_like(smoothed_index, dtype=np.int8)
    binned_map[mask & (smoothed_index < t1)] = 1
    binned_map[mask & (smoothed_index >= t1) & (smoothed_index < t2)] = 2
    binned_map[mask & (smoothed_index >= t2)] = 3
    
    # Map bins to strict RGBA values for overlay
    binned_rgba = np.zeros((*img_clahe.shape, 4))
    binned_rgba[binned_map == 1] = [*mcolors.to_rgb('#0072B2'), 0.6]
    binned_rgba[binned_map == 2] = [*mcolors.to_rgb('#F0E442'), 0.6]
    binned_rgba[binned_map == 3] = [*mcolors.to_rgb('#D55E00'), 0.6]

    fig_disc, axes_disc = plt.subplots(1, 2, figsize=(12, 6))
    axes_disc[0].imshow(img_clahe, cmap='gray')
    axes_disc[0].set_title('Normalized Input (CLAHE)')
    axes_disc[0].axis('off')
    
    axes_disc[1].imshow(img_clahe, cmap='gray')
    axes_disc[1].imshow(binned_rgba)
    axes_disc[1].set_title('Discrete Texture Bins')
    axes_disc[1].axis('off')
    
    # Constructing the Legend
    legend_elements = [
        mpatches.Patch(color='#0072B2', label=f'Tubular (< {t1})'),
        mpatches.Patch(color='#F0E442', label='Intermediate'),
        mpatches.Patch(color='#D55E00', label=f'Spotty (>= {t2})')
    ]
    # Place legend outside the image plot to avoid obscuring biology
    axes_disc[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{base_name}_comp_discrete.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_disc) 

    # --- Statistics ---
    total_mito_pixels = np.sum(mask)
    pixels_tubular = np.sum(binned_map == 1)
    pixels_interme = np.sum(binned_map == 2)
    pixels_spotty  = np.sum(binned_map == 3)

    return {
        "total_mito_px": total_mito_pixels,
        "tubular_px": pixels_tubular,
        "interme_px": pixels_interme,
        "spotty_px": pixels_spotty
    }


def _worker_process_field(args):
    """
    Top-level wrapper function for multiprocessing.
    Must be outside the main function so ProcessPoolExecutor can pickle it safely.
    """
    file_path, well_id, field_id, output_paths_dict, target_channel, nuclei_channel, sigma, bin_thresholds = args
    
    if well_id not in output_paths_dict:
        return None
        
    well_out_dir = output_paths_dict[well_id]
    base_filename = os.path.basename(file_path).split('.')[0]
    
    mtg_path = file_path
    hoechst_path = file_path.replace(f"d{target_channel}", f"d{nuclei_channel}")
    base_filename_hoechst = base_filename.replace(f"d{target_channel}", f"d{nuclei_channel}")
    
    cell_count = count_nuclei(hoechst_path, well_out_dir, base_filename_hoechst, min_nucleus_area=500)
    mito_stats = analyze_and_save(mtg_path, well_out_dir, base_filename, sigma=sigma, bin_thresholds=bin_thresholds)

    safe_cells = cell_count if cell_count > 0 else np.nan
    
    field_data = {
        "Well_ID": well_id,
        "Field": field_id,
        "Cell_Count": cell_count,
        "Total_Mito_Px_per_Cell": mito_stats["total_mito_px"] / safe_cells,
        "Tubular_Px_per_Cell": mito_stats["tubular_px"] / safe_cells,
        "Interme_Px_per_Cell": mito_stats["interme_px"] / safe_cells,
        "Spotty_Px_per_Cell": mito_stats["spotty_px"] / safe_cells
    }
    return field_data


def process_plate_directory(data_dir, output_paths_dict, target_channel="1", nuclei_channel="0", sigma=1.5, test_well=None, bin_thresholds=(0.33, 0.66)):
    search_pattern = os.path.join(data_dir, "*.[tT][iI][fF]*")
    all_files = glob.glob(search_pattern)
    pattern = re.compile(r'_([A-H]\d{2})f(\d{2})d(\d)') 
    
    mtg_files = []
    for file in all_files:
        match = pattern.search(file)
        if match and match.group(3) == target_channel:
            well_id = match.group(1)
            if test_well and well_id != test_well:
                continue
            mtg_files.append((file, well_id, match.group(2)))

    # Package tasks for concurrent execution
    tasks = []
    for file_path, well_id, field_id in mtg_files:
        tasks.append((file_path, well_id, field_id, output_paths_dict, target_channel, nuclei_channel, sigma, bin_thresholds))

    all_field_data = []
    
    # Cap CPU usage slightly below max to keep the system responsive
    max_workers = min(os.cpu_count(), len(tasks)) if os.cpu_count() else 4

    print(f"Starting parallel processing with {max_workers} CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_process_field, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Processing Fields"):
            try:
                result = future.result()
                if result is not None:
                    all_field_data.append(result)
            except Exception as exc:
                print(f"A field generated an exception: {exc}")

    # --- EXPORT TO CSV ---
    if not all_field_data:
        print("No valid data processed.")
        return

    parent_out_dir = os.path.dirname(output_paths_dict[all_field_data[0]["Well_ID"]])
    csv_path = os.path.join(parent_out_dir, "Plate_Normalized_Data_Per_Field.csv")
    
    headers = ["Well_ID", "Field", "Cell_Count", "Total_Mito_Px_per_Cell", "Tubular_Px_per_Cell", "Interme_Px_per_Cell", "Spotty_Px_per_Cell"]
    
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in all_field_data:
            writer.writerow(row)
            
    print(f"\nData exported to: {csv_path}")

    # --- AGGREGATE PER WELL & PLOT (Median Filtering) ---
    wells = sorted(list(set(row["Well_ID"] for row in all_field_data)))
    
    med_tubular, med_interme, med_spotty = [], [], []

    for well in wells:
        well_rows = [r for r in all_field_data if r["Well_ID"] == well]
        med_tubular.append(np.nanmedian([r["Tubular_Px_per_Cell"] for r in well_rows]))
        med_interme.append(np.nanmedian([r["Interme_Px_per_Cell"] for r in well_rows]))
        med_spotty.append(np.nanmedian([r["Spotty_Px_per_Cell"] for r in well_rows]))

    med_tubular = np.array(med_tubular)
    med_interme = np.array(med_interme)
    med_spotty = np.array(med_spotty)
    total_med = med_tubular + med_interme + med_spotty

    safe_total = np.where(total_med == 0, np.nan, total_med)
    pct_tubular = (med_tubular / safe_total) * 100
    pct_interme = (med_interme / safe_total) * 100
    pct_spotty = (med_spotty / safe_total) * 100

    # PLOT 1: Absolute Mass (Median)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(wells, med_tubular, color='#0072B2', edgecolor='white', label='Tubular')
    ax1.bar(wells, med_interme, bottom=med_tubular, color='#F0E442', edgecolor='white', label='Intermediate')
    ax1.bar(wells, med_spotty, bottom=med_tubular+med_interme, color='#D55E00', edgecolor='white', label='Spotty')

    ax1.set_ylabel('Median Mito Pixels per Cell', fontweight='bold')
    ax1.set_xlabel('Well ID', fontweight='bold')
    ax1.set_title('Absolute Mitochondrial Mass per Cell', fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')
    fig1.tight_layout()
    plot1_path = os.path.join(parent_out_dir, "Plate_Absolute_Mass_Per_Cell.png")
    fig1.savefig(plot1_path, dpi=300)
    plt.close(fig1)

    # PLOT 2: Relative Composition (%)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(wells, pct_tubular, color='#0072B2', edgecolor='white', label='Tubular')
    ax2.bar(wells, pct_interme, bottom=pct_tubular, color='#F0E442', edgecolor='white', label='Intermediate')
    ax2.bar(wells, pct_spotty, bottom=pct_tubular+pct_interme, color='#D55E00', edgecolor='white', label='Spotty')

    ax2.set_ylabel('Network Composition (%)', fontweight='bold')
    ax2.set_xlabel('Well ID', fontweight='bold')
    ax2.set_title('Relative Mitochondrial Morphology', fontweight='bold')
    ax2.set_ylim(0, 100) 
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45, ha='right')
    fig2.tight_layout()
    plot2_path = os.path.join(parent_out_dir, "Plate_Relative_Composition.png")
    fig2.savefig(plot2_path, dpi=300)
    plt.close(fig2)

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    base_directory = r"C:/Users/Thomas/Desktop/Temp local/General NQO2 Search/Morphology"
    data_path = r"C:/Users/Thomas/Vandria/Superlab - 4 NQO2 Biology/Mitophagy/Microscopy/04 - TMRM & MitoTracker/TCH_003_TMRM_MTG_HepG2/Plate 2/Colocalization.V4_03-31-26_02;44;49"
    
    out_dir, paths_dict = setup_output(base_directory, "Output masks", zero_pad=True)

    process_plate_directory(
        data_dir=data_path, 
        output_paths_dict=paths_dict, 
        target_channel="1", 
        sigma=0.5, 
        test_well=None,
        bin_thresholds=(0.4, 0.5)
    )