import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_patient_data(patient_id, data_dir):
    '''
    Load the scan and mask data for a patient

    Parameters:
    patient_id (str): The patient ID
    data_dir (str): The directory containing the data

    Returns:
    - scan_array (np.array): The scan data
    - mask_array (np.array): The mask data
    '''
    scan_file = f"/{patient_id}.nii"
    mask_file = f"/{patient_id}_mask.nii"

    scan_nii = nib.load(data_dir + scan_file)
    mask_nii = nib.load(data_dir + mask_file)

    scan_array = np.array(scan_nii.get_fdata())
    mask_array = np.array(mask_nii.get_fdata())

    return scan_array, mask_array

def create_subvolume(scan, mask):
    '''
    Create a subvolume containing the mask and surrounding tissue'

    Parameters:
    - scan (np.array): The scan data
    - mask (np.array): The mask data

    Returns:
    - subvolume (np.array): The subvolume containing the mask and surrounding tissue
    - mask_subvolume (np.array): The mask subvolume
    '''
    # Get the center of the mask
    z_indices, y_indices, x_indices = np.where(mask > 0)

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)

    # Extend the range by 30 voxels in x and y directions, 5 in z direction
    x_min_extended = max(0, x_min - 30)
    x_max_extended = min(scan.shape[0] - 1, x_max + 30)
    y_min_extended = max(0, y_min - 30)
    y_max_extended = min(scan.shape[1] - 1, y_max + 30)
    z_min_extended = max(0, z_min - 5)
    z_max_extended = min(scan.shape[2] - 1, z_max + 5)

    subvolume = scan[z_min_extended:z_max_extended+1,
                     y_min_extended:y_max_extended+1,
                     x_min_extended:x_max_extended+1]

    mask_subvolume = mask[z_min_extended:z_max_extended+1,
                          y_min_extended:y_max_extended+1,
                          x_min_extended:x_max_extended+1]

    return subvolume, mask_subvolume

def find_intensity_range(scan, mask):
    '''
    Find the intensity range of the scan within the mask

    Parameters:
    - scan (np.array): The scan data
    - mask (np.array): The mask data

    Returns:
    - min_intensity (float): The minimum intensity within the mask
    - max_intensity (float): The maximum intensity within the mask
    '''
    intensities = scan[mask > 0]
    if len(intensities) == 0:
        return 0, 0
    min_intensity = np.min(intensities)
    max_intensity = np.max(intensities)

    return min_intensity, max_intensity

def apply_thresholding(subvolume_scan, subvolume_mask):
    ''' 
    Apply thresholding to the subvolume scan data based on the intensity range of the mask

    Parameters:
    - subvolume_scan (np.array): The subvolume scan data
    - subvolume_mask (np.array): The subvolume mask data

    Returns:
    - threshold_mask (np.array): The thresholded mask
    - threshold (tuple): The intensity range used for threshold
    '''

        # Find intensity range of voxels inside the segmentation
    min_intensity, max_intensity = find_intensity_range(subvolume_scan, subvolume_mask)

        # Create a thresholded mask based on the intensity range
    threshold_mask = np.zeros_like(subvolume_scan)
    threshold_mask[(subvolume_scan >= min_intensity) & (subvolume_scan <= max_intensity)] = 1

    threshold = (min_intensity, max_intensity)

    return threshold_mask, threshold


def energy(voxel_values):
    '''
    Calculate the energy of the voxel values
    
    Parameters:
    - voxel_values (np.array): The voxel values
    
    Returns:
    - energy (float): The energy of the voxel values
    '''
    return np.sum(voxel_values**2)

def mad(voxel_values):
  '''
  Calculate the mean absolute deviation of the voxel values

  Parameters:
  - voxel_values (np.array): The voxel values

  Returns:
  - mad_value (float): The mean absolute deviation of the voxel
  '''
  
  mean_value = np.mean(voxel_values)
  mad_value = np.mean(np.abs(voxel_values - mean_value))
  return mad_value

def uniformity(voxel_values,num_bins):
    '''
    Calculate the uniformity of the voxel values

    Parameters:
    - voxel_values (np.array): The voxel values
    - num_bins (int): The number of bins to use for the histogram

    Returns:
    - uniformity (float): The uniformity of the voxel values
    '''
    hist, _ = np.histogram(voxel_values, bins=num_bins, density=False)

    # Normalize histogram
    p = hist / len(voxel_values)

    # Calculate uniformity
    return np.sum(p**2)

def normalize_intensity(voxel_data):
    '''
    Normalize the intensity values of the voxel data
    
    Parameters:
    - voxel_data (np.array): The voxel data
    
    Returns:
    - normalized_data (np.array): The normalized voxel data
    '''
    min_val = np.min(voxel_data)
    max_val = np.max(voxel_data)
    return (voxel_data - min_val) / (max_val - min_val)






def main(data_dir, output_dir,saveing_large = False):
    patient_ids = []
    for file in os.listdir(data_dir):
        if file.endswith('.nii') and not file.endswith('_mask.nii'):
            patient_id = file.split('.')[0]
            patient_ids.append(patient_id)



    patient_scans = []
    segmentation_masks = []

    subvolume_scans= []
    subvolume_masks = []

    x = []
    y = []
    z = []
    # Get the NIfTI files
    for patient_id in tqdm(patient_ids):

        # patient data
        scans, mask = load_patient_data(patient_id, data_dir)
        if saveing_large:
            patient_scans.append(scans)
            segmentation_masks.append(mask)

        # subvolume data
        subvolume_scan, subvolume_mask = create_subvolume(scans, mask)
        z_indices, y_indices, x_indices = np.where(mask > 0)

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        x.append(x_min)
        x.append(x_max)
        y.append(y_min)
        y.append(y_max)
        z.append(z_min)
        z.append(z_max)

        if saveing_large:
            subvolume_scans.append(subvolume_scan)
            subvolume_masks.append(subvolume_mask)

    print((min(x),max(x)))
    print((min(y),max(y)))
    print((min(z),max(z)))
    patient_ids = sorted(patient_ids)

    case_0 = load_patient_data(patient_ids[0], data_dir)
    print(case_0[0].shape, case_0[1].shape)       

    #Dice score calculation
    dice_scores = {}

    for patient_id in patient_ids:
        print(patient_id)
        scans, mask = load_patient_data(patient_id, data_dir)
        subvolume_scan, subvolume_mask = create_subvolume(scans, mask)
        threshold_mask, thresholds = apply_thresholding(subvolume_scan, subvolume_mask)
        intersection = np.sum(threshold_mask * subvolume_mask)
        dice_score = (2.0 * intersection) / (np.sum(threshold_mask) + np.sum(mask))
        dice_scores[patient_id] = dice_score
        print(f"Dice score: {dice_score}")



    patient_id = "case_0"
    #patient_id = "case_3"


    scans, mask = load_patient_data(patient_id, data_dir)
    subvolume_scan, subvolume_mask = create_subvolume(scans, mask)
    threshold_masks, thresholds = apply_thresholding(subvolume_scan, subvolume_mask)
    slice_idx = subvolume_scan.shape[2] // 2  # Middle slice
    scan_slice = subvolume_scan[:,:,slice_idx]
    mask_slice = subvolume_mask[:,:,slice_idx]
    threshold_masks_slice = threshold_masks[:,:,slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    #Original scan
    axes[0].imshow(scan_slice, cmap='gray')
    axes[0].set_title('Original Scan')
    axes[0].axis('off')

    # ground truth mask
    axes[1].imshow(scan_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap='viridis', alpha=0.5)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(scan_slice, cmap='gray')
    axes[2].imshow(threshold_masks_slice, cmap='viridis', alpha=0.5)
    axes[2].set_title('Threshold Mask')
    axes[2].axis('off')

    fig.tight_layout()
    plt.savefig(output_dir + '/threshold_mask.png')
    plt.show()

    labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    #Label 1: healthy

    #Label 2: lung tumor

    #Label 3: metastatic tumor

    intensity_ranges = []
    bins_count = []
    for patient_id in patient_ids:
        patient_num = patient_id.split('_')[1]
        diagnosis = labels_df.loc[labels_df['ID'] == int(patient_num), 'Diagnosis'].values

        if diagnosis != 1:
            scan, mask = load_patient_data(patient_id, data_dir)
            voxel_values = scan[mask > 0]
            intensities = normalize_intensity(voxel_values)

            min_int = np.min(intensities)
            max_int = np.max(intensities)
            intensity_ranges.append((min_int,max_int))

            bins = int(np.ceil((max_int - min_int) / (3.5 * np.std(intensities) / len(intensities)**(1/3))))
            bins_count.append(bins)

    bins_number = int(np.mean(bins_count))
    print(bins_number)

    features = {}
    for patient_id in tqdm(patient_ids):
        scans, mask = load_patient_data(patient_id, data_dir)
        patient_num = patient_id.split('_')[1]
        diagnosis = labels_df.loc[labels_df['ID'] == int(patient_num), 'Diagnosis'].values

        scan = normalize_intensity(scan)

        intensities = scans[mask > 0]

        energy_val = energy(intensities)
        mad_val = mad(intensities)
        uniformity_val = uniformity(intensities, num_bins=bins_number)
        features[patient_id] = [energy_val, mad_val, uniformity_val,diagnosis]

    #created a dataframe for all the statistics
    df = pd.DataFrame.from_dict(features, orient='index')
    df.columns = ['Energy','Mean Absolute Deviation','Uniformity', 'Diagnosis']
    df = df.round({'Energy': 2, 'Mean Absolute Deviation': 2, 'Uniformity': 2})
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Case_ID'}, inplace=True)
    df.to_csv(output_dir + '/features.csv', index=False)
    #print(df.head())