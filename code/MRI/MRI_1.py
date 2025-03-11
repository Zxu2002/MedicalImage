import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2


def kspace_to_image(kspace):
    '''
    This function converts k-space data to image space.

    Parameters:
    - kspace (numpy.ndarray): The k-space data.

    Returns:
    - image (numpy.ndarray): The image space data.
    '''
    image = np.fft.fftshift(kspace)
    image = np.fft.ifft2(image)
    
    
    return image

def main(data_path,output_path = "graph"):
    kspace = np.load(data_path + "/kspace.npy")
    print(kspace.shape)

    #2.1.1
    num_coils = kspace.shape[0]  

    # 2.1.2
    plt.figure(figsize=(15, 5))
    for i in range(num_coils):
        plt.subplot(2, 3, i+1)
        kspace_mag = np.log1p(np.abs(kspace[i]))
        plt.imshow(kspace_mag, cmap='gray')
        plt.title(f'Coil {i+1} k-space magnitude')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path + "/all_coil_kspace_magnitude.png")
    plt.show()

    #2.1.3
    image_data = np.zeros_like(kspace, dtype=complex)
    for i in range(num_coils):
        image_data[i] = kspace_to_image(kspace[i])

    coil_idx = 0  # First coil

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(image_data[coil_idx]), cmap='gray')
    plt.title(f'Coil {coil_idx+1} Magnitude')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(image_data[coil_idx]), cmap='gray')
    plt.title(f'Coil {coil_idx+1} Phase')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path + "/coil1_magnitude_phase.png")
    plt.show()

    #2.1.4
    plt.figure(figsize=(15, 5))
    for i in range(num_coils):
        plt.subplot(2, 3, i+1)
        plt.imshow(np.abs(image_data[i]), cmap='gray')
        plt.title(f'Coil {i+1} Magnitude')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path + "/all_coil_magnitude.png")
    plt.show()

    #2.1.5
    combined = np.zeros(image_data.shape[1:])
    for i in range(num_coils):
        combined += np.abs(image_data[i])**2
    combined = np.sqrt(combined)

    plt.figure(figsize=(8, 8))
    plt.imshow(combined, cmap='gray')
    plt.title('Combined Image (Sum of Squares)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path + "/combined_image.png")
    plt.show()