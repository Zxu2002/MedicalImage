import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import iradon, radon
from tqdm import tqdm
import os

# OS-SART implementation
def os_sart(sinogram, angles, n_iterations=100, gamma=0.01,n_subsets=10):
    '''
    This function performs the OS-SART algorithm for reconstruction of CT images from sinograms.

    Parameters:
    - sinogram (numpy.ndarray): The sinogram data.
    - angles (numpy.ndarray): The angles used for the sinogram acquisition.
    - n_iterations (int): The number of iterations to perform.
    - gamma (float): The relaxation parameter.
    - n_subsets (int): The number of subsets to use.

    Returns:
    - recon (numpy.ndarray): The reconstructed CT image.
    '''
    # Get dimensions
    n_angles = len(angles)
    subset_size = n_angles // n_subsets

    # recon = ct_fbp.copy()
    img_shape = int(sinogram.shape[0])
    recon = np.zeros((img_shape, img_shape))
    angle_subsets = []
    sino_subsets = []

    for i in range(n_subsets):
        start_idx = i * subset_size
        end_idx = min(start_idx + subset_size, n_angles)
        
        angle_subsets.append(angles[start_idx:end_idx])
        sino_subsets.append(sinogram[:, start_idx:end_idx])

    # Main loop
    for _ in tqdm(range(n_iterations)):
        for subset_idx in range(len(angle_subsets)):
      
            current_angles = angle_subsets[subset_idx]
            current_sino = sino_subsets[subset_idx]

            forward_proj = radon(recon, theta=current_angles)

            error_sino = current_sino - forward_proj
            
            update = iradon(error_sino, theta=current_angles, filter_name=None)

    
            recon += gamma * update


    return recon

def sirt(sinogram,angles,n_iterations = 100, gamma = 0.01):
    '''
    This function performs the SIRT algorithm for reconstruction of CT images from sinograms.
    
    Parameters:
    - sinogram (numpy.ndarray): The sinogram data.
    - angles (numpy.ndarray): The angles used for the sinogram acquisition.
    - n_iterations (int): The number of iterations to perform.
    - gamma (float): The relaxation parameter.

    Returns:
    - x (numpy.ndarray): The reconstructed CT image.
    '''
    img_shape = int(sinogram.shape[0])
    x = np.zeros((img_shape, img_shape))
    for _ in tqdm(range(n_iterations)):
        residual = sinogram-radon(x,angles)
        gradient = gamma*iradon(residual,angles,filter_name=None) 
        x = x + gradient
    return x

def main(data_path,output_path = "graph"):
    #1.1
    ct_sino=np.load(data_path + "/ct_sinogram.npy")
    pet_sino=np.load(data_path + "/pet_sinogram.npy")

    #Additional data provided for correction 
    ct_dark = np.load(data_path + "/ct_dark.npy")
    ct_flat = np.load(data_path + "/ct_flat.npy")
    pet_calibration = np.load(data_path + "/pet_calibration.npy")

    #visualize the sinograms
    plt.figure()
    plt.subplot(121)
    plt.imshow(pet_sino,cmap="gray_r") 
    plt.title("Original PET scan")
    plt.axis("equal")
    plt.subplot(122)
    plt.imshow(ct_sino, cmap="gray")
    plt.title("Origional CT scan")
    plt.axis("equal")
    plt.savefig(output_path + "/ct_pet.png")
    plt.show()

    #Compute the corrected sinograms
    ct_corrected = -np.log((ct_sino - ct_dark) / (ct_flat - ct_dark))
    pet_corrected = pet_sino / pet_calibration

    #visualize the corrected sinograms
    plt.figure()
    plt.subplot(121)
    plt.imshow(pet_corrected,cmap="gray_r")
    plt.title("Corrected PET scan")
    plt.axis("equal")   
    plt.subplot(122)
    plt.imshow(ct_corrected, cmap="gray")
    plt.title("Corrected CT scan")
    plt.axis("equal")
    plt.savefig(output_path + "/ct_pet_corrected.png")
    plt.show()

    #1.2

    # FBP reconstruction
    #print(ct_corrected.shape)
    ct_fbp = iradon(ct_corrected, theta=np.linspace(0, 180, ct_corrected.shape[1]),filter_name = "ramp")



    # Run OS-SART reconstruction with your specific sinogram dimensions (512, 180)
    angles = np.linspace(0, 180, 180,endpoint=False)

    ct_ossart = os_sart(ct_corrected, angles,n_iterations = 100, gamma = 0.001)
    os.makedirs("saved_data", exist_ok=True)
    np.save("saved_data/ct_ossart.npy",ct_ossart)
    ct_sirt = sirt(ct_corrected,angles,n_iterations = 100, gamma = 0.001)

    n_iter = [50,100,150]
    gamma = [0.001,0.005,0.01]
    image_ind = 1
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    for i in n_iter:
        for j in gamma:
            plt.subplot(int(f"{len(n_iter)}{len(gamma)}{image_ind}"))
            ct_ossart_plot = os_sart(ct_corrected, angles,n_iterations = i, gamma = j)
            plt.imshow(ct_ossart_plot, cmap="gray")
            plt.title(f"K = {i} $\gamma$ = {j}")
            plt.axis("equal")
            image_ind += 1

    plt.savefig(output_path + "/ct_ossart_params.png")
    plt.show()
    # Display results
    #FBP vs OS-SART
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(ct_fbp, cmap="gray")
    plt.title("FBP Reconstruction")
    plt.axis("equal")

    plt.subplot(122)
    plt.imshow(ct_ossart, cmap="gray")
    plt.title("OS-SART Reconstruction")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path + "/ct_recon_sart_fbp.png")
    plt.show()


    #SIRT vs OS-SART
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(ct_sirt, cmap="gray")
    plt.title("SIRT Reconstruction")
    plt.axis("equal")

    plt.subplot(122)
    plt.imshow(ct_ossart, cmap="gray")
    plt.title("OS-SART Reconstruction")
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path + "/ct_recon_sart_sirt.png")
    plt.show()

