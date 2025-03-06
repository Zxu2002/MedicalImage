import numpy as np 
import matplotlib.pyplot as plt
from skimage.transform import iradon, radon, resize
from scipy.ndimage import zoom
from tqdm import tqdm
#1.1
ct_sino=np.load("code/Module1/ct_sinogram.npy")
pet_sino=np.load("code/Module1/pet_sinogram.npy")

#Additional data provided for correction 
ct_dark = np.load("code/Module1/ct_dark.npy")
ct_flat = np.load("code/Module1/ct_flat.npy")
pet_calibration = np.load("code/Module1/pet_calibration.npy")

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
plt.savefig("graphs/ct_pet.png")
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
plt.savefig("graphs/ct_pet_corrected.png")
plt.show()

#1.2

# FBP reconstruction
#print(ct_corrected.shape)
ct_fbp = iradon(ct_corrected, theta=np.linspace(0, 180, ct_corrected.shape[1]),filter_name = "ramp")

# OS-SART implementation
def os_sart(sinogram, angles, n_iterations=100, gamma=0.01):
    # Get dimensions
    n_angles = len(angles)
    n_subsets = 10  

    subset_size = n_angles // n_subsets

    # recon = ct_fbp.copy()
    recon = np.zeros_like(ct_fbp)
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
    x = np.zeros(ct_fbp.shape)
    for _ in tqdm(range(n_iterations)):
        residual = sinogram-radon(x,angles)
        gradient = gamma*iradon(residual,angles,filter_name=None) 
        x = x + gradient
    return x

# Run OS-SART reconstruction with your specific sinogram dimensions (512, 180)
angles = np.linspace(0, 180, 180,endpoint=False)

ct_ossart = os_sart(ct_corrected, angles,n_iterations = 100, gamma = 0.001)
np.save("code/Module1/ct_ossart.npy",ct_ossart)
ct_sirt = sirt(ct_corrected,angles,n_iterations = 100, gamma = 0.001)
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
plt.savefig("graphs/ct_recon_sart_fbp.png")
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
plt.savefig("graphs/ct_recon_sart_sirt.png")
plt.show()

