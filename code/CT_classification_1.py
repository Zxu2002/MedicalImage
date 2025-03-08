import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path('Module3')

    patient_scans = []
    segmentation_masks = []
    patient_ids = []

    # Get the NIfTI files 
    scan_files = sorted([f for f in os.listdir(Path('Module3')) if f.endswith('.nii') and not f.endswith('_mask.nii')])
    mask_files = sorted([f for f in os.listdir(Path('Module3')) if f.endswith('_mask.nii')])

    for scan_file, mask_file in zip(scan_files, mask_files):
        # Extract patient ID for reference
        patient_id = scan_file.split('.nii')[0]
        patient_ids.append(patient_id)
        
        # Load the scan and mask using nibabel
        scan_nii = nib.load(data_dir / scan_file)
        mask_nii = nib.load(data_dir / mask_file)
        
        # Convert to numpy arrays
        scan_array = scan_nii.get_fdata()
        mask_array = mask_nii.get_fdata()
        
        # Store in our lists
        patient_scans.append(scan_array)
        segmentation_masks.append(mask_array)
        
        print(f"Loaded patient {patient_id}, scan shape: {scan_array.shape}, mask shape: {mask_array.shape}")
