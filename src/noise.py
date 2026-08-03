import numpy as np

def add_kspace_noise(k_space, noise_std, seed=None):
    if noise_std <= 0:
        return k_space
        
    rng = np.random.default_rng(seed)
    component_std = noise_std / np.sqrt(2.0)
    
    noise = (rng.normal(0, component_std, k_space.shape)
             + 1j * rng.normal(0, component_std, k_space.shape))
             
    return k_space + noise
