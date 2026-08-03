import numpy as np
from src.noise import add_kspace_noise


def apply_undersampling(image_data, acceleration_factor, noise_std=0.0, seed=None):
    k_space = np.fft.fftshift(np.fft.fft2(image_data))
    k_space = add_kspace_noise(k_space, noise_std, seed=seed)

    rows, cols = k_space.shape
    crow = rows // 2

    mask = np.zeros((rows, cols))

    start_row = crow % acceleration_factor
    mask[start_row::acceleration_factor, :] = 1

    center_region = 10
    
    low_bound = max(0, crow - center_region)
    high_bound = min(rows, crow + center_region)
    mask[low_bound:high_bound, :] = 1

    undersampled_k_space = k_space * mask * acceleration_factor

    return undersampled_k_space
