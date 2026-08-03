import numpy as np


def _coil_geometry(rows, cols, num_coils, ring_radius_factor=0.62):
    crow, ccol = rows / 2.0, cols / 2.0
    ring_radius = ring_radius_factor * max(rows, cols)
    thetas = np.linspace(0, 2 * np.pi, num_coils, endpoint=False)
    centers_y = crow + ring_radius * np.sin(thetas)
    centers_x = ccol + ring_radius * np.cos(thetas)
    return list(zip(centers_y, centers_x)), ring_radius


def coil_positions(rows, cols, num_coils, ring_radius_factor=0.62):
    return _coil_geometry(rows, cols, num_coils, ring_radius_factor)


def simulate_coils(image_data, num_coils=8, standoff=None, noise_std=0.0,
                    object_phase=True, seed=None):
    rng = np.random.default_rng(seed)
    rows, cols = image_data.shape

    centers, ring_radius = _coil_geometry(rows, cols, num_coils)
    if standoff is None:
        standoff = 0.35 * (2 * np.pi * ring_radius / num_coils)

    y, x = np.mgrid[0:rows, 0:cols].astype(float)

    if object_phase:
        yy, xx = np.mgrid[-1:1:rows * 1j, -1:1:cols * 1j]
        phase_map = 0.6 * (xx ** 2 - yy ** 2) + 0.3 * xx * yy
        complex_object = image_data * np.exp(1j * phase_map)
    else:
        complex_object = image_data.astype(complex)

    sensitivities = []
    for cy, cx in centers:
        dy = y - cy
        dx = x - cx
        r = np.sqrt(dx ** 2 + dy ** 2 + standoff ** 2)
        r = np.clip(r, standoff * 0.5, None) 
        
        C = 3 * (dy * standoff + 1j * dx * standoff) / (r ** 5)
        sensitivities.append(C)

    peak = max(np.max(np.abs(s_map)) for s_map in sensitivities)
    sensitivities = [C / peak for C in sensitivities]

    coils = []
    for C in sensitivities:
        coil_image = complex_object * C
        if noise_std > 0:
            noise = (rng.normal(0, noise_std, coil_image.shape)
                      + 1j * rng.normal(0, noise_std, coil_image.shape))
            coil_image = coil_image + noise
        coils.append(coil_image)

    return coils, sensitivities


def root_sum_of_squares(coils):
    stacked = np.stack(coils, axis=0)
    return np.sqrt(np.sum(np.abs(stacked) ** 2, axis=0))


def optimal_combine(coils, sensitivities):
    stacked_c = np.stack(coils, axis=0)
    stacked_s = np.stack(sensitivities, axis=0)
    numerator = np.sum(np.conj(stacked_s) * stacked_c, axis=0)

    denominator = np.sum(np.abs(stacked_s) ** 2, axis=0)
    denominator = np.clip(denominator, 1e-12, None)
    
    return numerator / denominator
