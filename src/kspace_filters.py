import numpy as np

def apply_low_pass(k_space, radius):
    k_space_shifted = np.fft.fftshift(k_space)
    
    rows, cols = k_space.shape
    crow, ccol = rows // 2, cols // 2
    
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    
    mask = np.zeros((rows, cols))
    mask[mask_area] = 1
    
    filtered_shifted = k_space_shifted * mask
    return np.fft.ifftshift(filtered_shifted)

def apply_high_pass(k_space, radius):
    k_space_shifted = np.fft.fftshift(k_space)
    
    rows, cols = k_space.shape
    crow, ccol = rows // 2, cols // 2
 
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    
    mask = np.ones((rows, cols))
    mask[mask_area] = 0
    
    filtered_shifted = k_space_shifted * mask
    return np.fft.ifftshift(filtered_shifted)
