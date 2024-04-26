import os
import numpy as np
from PIL import Image

def convert_npy_to_tiff(npy_dir):
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
    
    for npy_file in npy_files:
        # load .npy file
        data = np.load(os.path.join(npy_dir, npy_file))
        
        # convert to PIL Image
        image = Image.fromarray(data)
        
        # aave as .tiff
        tiff_path = os.path.splitext(os.path.join(npy_dir, npy_file))[0] + '.tiff'
        image.save(tiff_path)
        print(f"Saved {tiff_path}")

npy_directory = r'C:\Users\axiong\Desktop\SVSTEST\20227001_tiles_numpy\tile_0_0_segmentation'
convert_npy_to_tiff(npy_directory)

