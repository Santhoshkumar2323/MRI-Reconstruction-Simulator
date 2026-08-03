import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from src.kspace_filters import apply_low_pass, apply_high_pass
from src.multi_coil import simulate_coils, root_sum_of_squares, optimal_combine, coil_positions
from src.undersample import apply_undersampling
from src.noise import add_kspace_noise

st.set_page_config(page_title="MRI Reconstruction Simulator", layout="wide")
st.title("Let's turn raw MRI data into images")

SCAN_DIR = "sample_scans"
if not os.path.exists(SCAN_DIR):
    os.makedirs(SCAN_DIR)

available_scans = [f for f in os.listdir(SCAN_DIR) if f.endswith('.npy')]


st.sidebar.header("Control Panel")

if not available_scans:
    st.sidebar.error(f"No .npy files found in the '{SCAN_DIR}' folder. Please add some to proceed.")
else:
    selected_scan = st.sidebar.selectbox("Select Patient Scan", available_scans)

    experiment_type = st.sidebar.radio(
        "Select Experiment",
        ("K-Space Filters", "Multi-Coil Merge (RSS)", "Undersampling (Aliasing)")
    )

    noise_std = st.sidebar.slider(
        "Thermal Noise (k-space std)", 0.0, 0.15, 0.02, step=0.005,
        help="Complex Gaussian noise added directly to k-space, mimicking real "
             "receiver/coil thermal (Johnson-Nyquist) noise."
    )

    image_data = np.load(os.path.join(SCAN_DIR, selected_scan))

    if image_data.ndim > 2:
        image_data = image_data[:, :, 0] 
    image_data = image_data.astype(float)

    col1, col2 = st.columns(2)

    if experiment_type == "K-Space Filters":
        st.sidebar.subheader("Filter Settings")
        filter_type = st.sidebar.radio("Filter", ("Full Scan", "Low-Pass (Contrast)", "High-Pass (Edges)"))

        max_rad = min(image_data.shape) // 2
        radius = st.sidebar.slider("Filter Radius", min_value=5, max_value=max_rad, value=30)

        k_space = np.fft.fft2(image_data)
        k_space = add_kspace_noise(k_space, noise_std)


        if filter_type == "Full Scan":
            masked_k_space = k_space
        elif filter_type == "Low-Pass (Contrast)":
            masked_k_space = apply_low_pass(k_space.copy(), radius)
        elif filter_type == "High-Pass (Edges)":
            masked_k_space = apply_high_pass(k_space.copy(), radius)


        reconstructed = np.abs(np.fft.ifft2(masked_k_space))

        with col1:
            st.subheader("Physics (K-Space)")
            fig, ax = plt.subplots()
            shifted_vis = np.fft.fftshift(masked_k_space)
            ax.imshow(np.log(1 + np.abs(shifted_vis)), cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("Anatomy (Reconstruction)")
            fig, ax = plt.subplots()
            ax.imshow(reconstructed, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

    elif experiment_type == "Multi-Coil Merge (RSS)":
        st.sidebar.subheader("Array Settings")
        num_coils = st.sidebar.slider("Number of Coil Elements", 4, 16, 8, step=2)

        coils, sensitivities = simulate_coils(
            image_data, num_coils=num_coils, noise_std=noise_std, object_phase=True
        )

        rows, cols = image_data.shape
        centers, ring_radius = coil_positions(rows, cols, num_coils)

        st.subheader("Array Geometry")
        fig, ax = plt.subplots()
        ax.imshow(image_data, cmap='gray')
        cy = [c[0] for c in centers]
        cx = [c[1] for c in centers]
        ax.scatter(cx, cy, c='red', s=60, marker='s', label='Coil elements')
        padding = 40
        ax.set_xlim(-padding, cols + padding)
        ax.set_ylim(rows + padding, -padding)

        ax.axis('off')
        ax.legend(loc='upper right')
        st.pyplot(fig)
        st.caption(
            "Each coil is a loop element on a ring around the anatomy, like a real body/head "
            "phased-array. Sensitivities below come from the actual off-axis magnetic dipole "
            "field of a current loop (Biot-Savart law), not a hand-tuned falloff curve."
        )

        n_show = min(len(sensitivities), 8)

        st.subheader("Per-Coil Sensitivity Maps (magnitude)")
        cols_grid = st.columns(n_show)
        for i, (col, S) in enumerate(zip(cols_grid, sensitivities[:n_show])):
            with col:
                fig, ax = plt.subplots()
                ax.imshow(np.abs(S), cmap='viridis')
                ax.axis('off')
                ax.set_title(f"Coil {i + 1}", fontsize=9)
                st.pyplot(fig)

        st.subheader("Per-Coil Images (magnitude)")
        cols_grid = st.columns(n_show)
        for i, (col, coil_img) in enumerate(zip(cols_grid, coils[:n_show])):
            with col:
                fig, ax = plt.subplots()
                ax.imshow(np.abs(coil_img), cmap='gray')
                ax.axis('off')
                ax.set_title(f"Coil {i + 1}", fontsize=9)
                st.pyplot(fig)

        rss_image = root_sum_of_squares(coils)
        combined_complex = optimal_combine(coils, sensitivities)

        st.subheader("Combined Reconstruction")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**RSS (magnitude-only)**")
            fig, ax = plt.subplots()
            ax.imshow(rss_image, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        with c2:
            st.markdown("**Sensitivity-Weighted Combine (magnitude)**")
            fig, ax = plt.subplots()
            ax.imshow(np.abs(combined_complex), cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
        with c3:
            st.markdown("**Sensitivity-Weighted Combine (phase)**")
            fig, ax = plt.subplots()
            ax.imshow(np.angle(combined_complex), cmap='twilight')
            ax.axis('off')
            st.pyplot(fig)
        st.caption(
            "RSS throws away phase entirely. The sensitivity-weighted (Roemer) combine recovers "
            "the true object phase, shown at right -- this is the same principle SENSE/GRAPPA "
            "reconstructions depend on."
        )

    elif experiment_type == "Undersampling (Aliasing)":
        st.sidebar.subheader("Speed Settings")
        acceleration = st.sidebar.slider("Acceleration Factor", min_value=2, max_value=8, value=2)

        undersampled_k_space = apply_undersampling(image_data, acceleration, noise_std=noise_std)

        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(undersampled_k_space)))

        with col1:
            st.subheader("Physics (Missing Data)")
            fig, ax = plt.subplots()
            ax.imshow(np.log(1 + np.abs(undersampled_k_space)), cmap='gray')
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("Anatomy (Aliasing Artifacts)")
            fig, ax = plt.subplots()
            ax.imshow(reconstructed, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
