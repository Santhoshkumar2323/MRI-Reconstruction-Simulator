import os
import shutil
import numpy as np
import pydicom
from tcia_utils import nbia


TARGET_MATH_DIR = "sample_scans"     
TEMP_DOWNLOADS = "tciaDownload"      

os.makedirs(TARGET_MATH_DIR, exist_ok=True)

selected_series = [
    "1.3.6.1.4.1.14519.5.2.1.186749128823666050588710146976747922803",
    "1.3.6.1.4.1.14519.5.2.1.223540088060091256676039744747833703676",
    "1.3.6.1.4.1.14519.5.2.1.3485478800445345800656575785739835751"
]

MAX_TOTAL_SLICES = 30
saved_count = 0

print(f"🚀 Contacting official TCIA databases to extract {MAX_TOTAL_SLICES} real MRI slices...")

for idx, series_id in enumerate(selected_series, start=1):
    if saved_count >= MAX_TOTAL_SLICES:
        break
        
    print(f"\n📥 Downloading Series Volume {idx}/{len(selected_series)}...")
    try:
        if os.path.exists(TEMP_DOWNLOADS):
            shutil.rmtree(TEMP_DOWNLOADS)
            
        nbia.downloadSeries(
            [series_id], 
            input_type="list"
        )
        
        dicom_files = []
        for root, _, files in os.walk(TEMP_DOWNLOADS):
            for file in files:
                if file.lower().endswith('.dcm') or not '.' in file:
                    dicom_files.append(os.path.join(root, file))
                    
        if not dicom_files:
            print("⚠️ No medical image instances extracted from this run.")
            continue
            
        print(f"📦 Found {len(dicom_files)} slices. Converting data...")
        
        for file_path in dicom_files:
            if saved_count >= MAX_TOTAL_SLICES:
                break
                
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'pixel_array'):
                    matrix = ds.pixel_array.astype(np.float32)
                    
                    saved_count += 1
                    filename = f"mri_slice_{saved_count:02d}.npy"
                    output_path = os.path.join(TARGET_MATH_DIR, filename)
                    
                    np.save(output_path, matrix)
                    print(f" Matrix Written -> {output_path} (Shape: {matrix.shape})")
            except Exception:
                continue
                
    except Exception as e:
        print(f"❌ Failed processing sequence run {idx}: {str(e)}")

if os.path.exists(TEMP_DOWNLOADS):
    shutil.rmtree(TEMP_DOWNLOADS)

print(f"\n Pipeline finished successfully! Generated {saved_count} matrices in '{TARGET_MATH_DIR}/'.")
