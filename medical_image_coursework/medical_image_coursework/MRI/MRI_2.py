import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
from scipy.ndimage import gaussian_filter
import cv2
from skimage.restoration import denoise_wavelet


def butterworth_lowpass_filter(shape, D0=30, n=2): 
    '''
    This function creates a Butterworth lowpass filter.

    Parameters:
    - shape (tuple): The shape of the filter.
    - D0 (float): The cutoff frequency.
    - n (int): The order of the filter.

    Returns:
    - H (numpy.ndarray): The Butterworth lowpass filter.

    '''
    P, Q = shape[0], shape[1]
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    U, V = np.meshgrid(u, v, indexing='ij') 
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / D0) ** (2 * n)) 
    return H

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



if __name__ == "__main__":
    # Load k-space data
    kspace = np.load("code/Module2/kspace.npy")
    print(kspace.shape)

    # convert k-space data to image 
    image_data = np.zeros_like(kspace, dtype=complex)
    combined = np.zeros(image_data.shape[1:])
    for i in range(kspace.shape[0]):
        image_data[i] = kspace_to_image(kspace[i])
        combined += np.abs(image_data[i])**2

        
    combined = np.sqrt(combined)
    # image_data = combined


    #2.2.1
    gaussian_filtered = np.zeros_like(image_data)
    bilateral_filtered = np.zeros_like(image_data)
    wavelet_filtered = np.zeros_like(image_data)

    gaussian_filtered_combine = np.zeros_like(combined)
    bilateral_filtered_combine = np.zeros_like(combined)
    wavelet_filtered_combine = np.zeros_like(combined)
    for i in range(image_data.shape[0]):  
        img_mag = np.abs(image_data[i])
        # 1. Gaussian filtering
        gaussian_filtered[i] = gaussian_filter(img_mag, sigma=1.0)
        gaussian_filtered_combine += np.abs(gaussian_filtered[i])**2


        # 2. Bilateral filtering
        bilateral_filter = cv2.bilateralFilter(img_mag.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)
        bilateral_filtered[i] = bilateral_filter 
        bilateral_filtered_combine += np.abs(bilateral_filter)**2

        # 3. Wavelet denoising
        wavelet_filtered[i] = denoise_wavelet(img_mag, method="BayesShrink", mode="soft", rescale_sigma=True)
        wavelet_filtered_combine += np.abs(wavelet_filtered[i])**2
    
    gaussian_filtered_combine = np.sqrt(gaussian_filtered_combine)
    bilateral_filtered_combine = np.sqrt(bilateral_filtered_combine)
    wavelet_filtered_combine = np.sqrt(wavelet_filtered_combine)
                                            

    # Visualize results for all coils
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(combined, cmap='gray')
    axes[0, 0].set_title('Original Combined Image')
    axes[0, 0].axis('off')
    
    # Gaussian filtered
    axes[0, 1].imshow(gaussian_filtered_combine, cmap='gray')
    axes[0, 1].set_title('Gaussian Filtered')
    axes[0, 1].axis('off')
    
    # Bilateral filtered
    axes[1, 0].imshow(bilateral_filtered_combine, cmap='gray')
    axes[1, 0].set_title('Bilateral Filtered')
    axes[1, 0].axis('off')
    
    # Wavelet filtered
    axes[1, 1].imshow(wavelet_filtered_combine, cmap='gray')
    axes[1, 1].set_title('Wavelet Filtered')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


    #2.2.2
    coil_idx = 0
    kspace_coil = kspace[coil_idx]

    # Create the filter (match dimensions of k-space data)
    butter_filter = butterworth_lowpass_filter(kspace_coil.shape, D0=30, n=2)
    filtered_kspace = kspace_coil * butter_filter

    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_kspace))

    # Display the original and filtered images (magnitude and phase)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Original magnitude and phase
    axes[0, 0].imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_coil))), cmap='gray')
    axes[0, 0].set_title('Original Magnitude')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.angle(np.fft.ifft2(np.fft.ifftshift(kspace_coil))), cmap='hsv')
    axes[0, 1].set_title('Original Phase')
    axes[0, 1].axis('off')

    # Filtered magnitude and phase
    axes[1, 0].imshow(np.abs(filtered_image), cmap='gray')
    axes[1, 0].set_title('Butterworth Filtered Magnitude')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.angle(filtered_image), cmap='hsv')
    axes[1, 1].set_title('Butterworth Filtered Phase')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("graph/Butterworth_Filtered.png")
    plt.show()

    #2.2.3
    butter_image = np.zeros_like(image_data)
    butter_combine = np.zeros_like(combined)
    for i in range(kspace.shape[0]):
        filtered_kspace = kspace[i] * butter_filter
        filtered_image = kspace_to_image(filtered_kspace)
        butter_image[i] = filtered_image
        butter_combine += np.abs(filtered_image)**2
    
    butter_combine = np.sqrt(butter_combine)

    fig,axes = plt.subplots(1,2,figsize=(12,10))
    axes[0].imshow(combined,cmap="gray")
    axes[0].set_title("Original Combined Image")
    axes[0].axis("off")
    axes[1].imshow(butter_combine,cmap="gray")
    axes[1].set_title("Butterworth Filtered")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
