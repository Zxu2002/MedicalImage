#1.3
import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import iradon, radon, resize
from tqdm import tqdm


def MLEM(corrected_pet_sino, angles, num_iterations):
    '''
    This function performs the MLEM algorithm for reconstruction of PET images from sinograms.

    Parameters:
    - corrected_pet_sino (numpy.ndarray): The corrected PET sinogram.
    - angles (numpy.ndarray): The angles used for the sinogram acquisition.
    - num_iterations (int): The number of iterations to perform.

    Returns:
    - mlem_reconstruction (numpy.ndarray): The reconstructed PET image.
    '''
    pet_detector_bins = corrected_pet_sino.shape[0] 
    mlem_reconstruction = np.ones((pet_detector_bins, pet_detector_bins))
    
    for iteration in tqdm(range(num_iterations)):
        forward_proj = radon(mlem_reconstruction, theta=angles, circle=True)
        
        ratio = corrected_pet_sino / (forward_proj + 1e-10)
        
        backproj_ratio = iradon(ratio, theta=angles, circle=True, filter_name=None)
        
        mlem_reconstruction *= backproj_ratio
    
    return mlem_reconstruction

def OSEM(corrected_pet_sino, angles, num_iterations, num_subsets, subset_indices, subset_angles):
    '''
    This function performs the OSEM algorithm for reconstruction of PET images from sinograms.
    
    Parameters:
    - corrected_pet_sino (numpy.ndarray): The corrected PET sinogram.
    - angles (numpy.ndarray): The angles used for the sinogram acquisition.
    - num_iterations (int): The number of iterations to perform.
    - num_subsets (int): The number of subsets to use.
    - subset_indices (list): The indices of the subsets.
    - subset_angles (list): The angles for each subset.

    Returns:
    - osem_reconstruction (numpy.ndarray): The reconstructed PET image.
    '''
    pet_detector_bins = corrected_pet_sino.shape[0] 
    osem_reconstruction = np.ones((pet_detector_bins, pet_detector_bins))
    
    for iteration in tqdm(range(num_iterations)):
        for subset in range(num_subsets):
            # Get current subset angles and sinogram data
            current_angles = angles[subset_indices[subset]]  # Use the correct angles from original array
            current_sinogram = corrected_pet_sino[:, subset_indices[subset]]  # Select columns for this subset
            
            # Forward projection - note the shape
            forward_proj = radon(osem_reconstruction, theta=current_angles, circle=True)

            ratio = current_sinogram / (forward_proj + 1e-10)
            
            backproj_ratio = iradon(ratio, theta=current_angles, circle=True, filter_name=None)

            osem_reconstruction *= backproj_ratio
    
    return osem_reconstruction


def main(data_path,output_path = "graph"):
    #loads the data 
    ct_sino=np.load(data_path + "/ct_sinogram.npy")
    pet_sino=np.load(data_path + "/pet_sinogram.npy")

    #Additional data provided for correction 
    ct_dark = np.load(data_path + "/ct_dark.npy")
    ct_flat = np.load(data_path + "/ct_flat.npy")
    pet_calibration = np.load(data_path + "/pet_calibration.npy")

    # Perform the corrections
    ct_corrected = -np.log((ct_sino - ct_dark) / (ct_flat - ct_dark))
    pet_corrected = pet_sino / pet_calibration

    print(f"CT sinogram shape: {ct_corrected.shape}")
    print(f"PET sinogram shape: {pet_sino.shape}")

    # Visualize the corrected sinograms
    ct_image = iradon(ct_corrected, theta=np.linspace(0, 180, ct_corrected.shape[1]),filter_name = "ramp")
    print(f"CT image shape: {ct_image.shape}")
    ct_image = np.load("saved_data/ct_ossart.npy")
    ct_shape = ct_image.shape


    ct_pixel_size = 1.06  
    pet_pixel_size = 4.24 
    scale_factor =  ct_pixel_size / pet_pixel_size


    target_size = int(ct_shape[0] * scale_factor) 
    resized_ct = resize(ct_image, (target_size, target_size), order=1, mode='reflect')
    # print(f"Resized CT image shape: {resized_ct.shape}")

    # Generate the PET angles
    pet_angles = pet_sino.shape[0]
    pet_detector_bins = pet_sino.shape[1]
    angle_pet = np.linspace(0., 180., pet_detector_bins, endpoint=False)


    ct_pet_sino = radon(resized_ct, theta=angle_pet, circle=True)
    # print(f"CT sinogram for PET shape: {ct_pet_sino.shape}")


    attenuation_correction = np.exp(ct_pet_sino)


    corrected_pet_sino = pet_sino * attenuation_correction

    # Visualize results
    #Attenuation correction
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(pet_sino, cmap='gray_r')
    plt.title('Original PET Sinogram')
    plt.colorbar()


    plt.subplot(132)
    plt.imshow(attenuation_correction, cmap='viridis')
    plt.title('Attenuation Correction Map')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(corrected_pet_sino, cmap='gray_r')
    plt.title('Attenuation-Corrected PET Sinogram')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + '/attenuation_correction_results.png')
    plt.show()




    angle_pet = np.linspace(0., 180., pet_detector_bins, endpoint=False)
    print(f"CT sinogram shape: {corrected_pet_sino.shape}")


    # FBP reconstruction
    fbp_reconstruction = iradon(corrected_pet_sino, theta=angle_pet, circle=True)
    # OSEM iterations
    print(corrected_pet_sino.shape)


    num_subsets = 10
    num_iterations = 1000
    angles_per_subset = pet_detector_bins // num_subsets
    subset_indices = [np.arange(i * angles_per_subset, (i + 1) * angles_per_subset) for i in range(num_subsets)]
    angle_pet = np.linspace(0., 180., pet_detector_bins, endpoint=False)


    osem_reconstruction = OSEM(corrected_pet_sino, angle_pet, num_iterations, num_subsets, subset_indices, subset_indices)
    mlem_reconstruction = MLEM(corrected_pet_sino, angle_pet, num_iterations)

    osem_reconstruction_100 = OSEM(corrected_pet_sino, angle_pet, 100, num_subsets, subset_indices, subset_indices)
    mlem_reconstruction_100 = MLEM(corrected_pet_sino, angle_pet, 100)

    # Display results

    #FBP vs OSEM
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(fbp_reconstruction, cmap='gray_r')
    plt.title('FBP Reconstruction')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(osem_reconstruction, cmap='gray_r')
    plt.title(f'OSEM Reconstruction ({num_iterations} iterations)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + '/pet_reconstruction_comparison.png')
    plt.show()


    #OSEM vs MLEM
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(osem_reconstruction, cmap='gray_r')
    plt.title('OSEM Reconstruction')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(mlem_reconstruction, cmap='gray_r')
    plt.title(f'MLEM Reconstruction ({num_iterations} iterations)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + '/pet_reconstruction_comparison_os_ml.png')
    plt.show()

    #OSEM vs MLEM 100 iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(osem_reconstruction_100, cmap='gray_r')
    plt.title('OSEM Reconstruction (100 iterations)')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(mlem_reconstruction_100, cmap='gray_r')
    plt.title(f'MLEM Reconstruction (100 iterations)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + '/pet_reconstruction_comparison_os_ml_100.png')
    plt.show()

    #Overlay
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(osem_reconstruction, cmap='gray_r')
    plt.title('OSEM Reconstruction')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(resized_ct, cmap='gray')
    plt.title('CT Reconstruction')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(osem_reconstruction, cmap='hot')
    plt.imshow(resized_ct, cmap='gray_r', alpha=0.6)
    plt.title('Overlay')
    plt.colorbar()
    plt.savefig(output_path + '/overlay_reconstruction.png')
    plt.show()
