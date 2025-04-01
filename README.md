# CMG Image Processing Scripts

This repository contains a collection of Python scripts developed to support preprocessing and segmentation of CMG and SVS microscopy image files. These scripts were written as part of a research co-op at BC Cancer, supporting deep learning workflows for biomedical image analysis.

## Contents

- `convert_jpg_to_tiff.py`  
  Converts JPEG images into TIFF format for standardization and downstream processing.

- `convert_npy_to_tiff.py`  
  Converts NumPy arrays into TIFF images for visualization or storage.

- `cropcmg.py`  
  Crops CMG binary image files into standardized segments for training and analysis.

- `openSlide.py`  
  Utility script for reading whole-slide SVS files using the OpenSlide C library.

- `segmentation_for_svs.ipynb`  
  Jupyter notebook performing segmentation on SVS files, used for preliminary analysis and model testing.

- `svs.py`  
  Script for parsing and segmenting SVS image files into labeled components.

## Technologies Used

- **Python 3**
- **NumPy**
- **OpenSlide-Python**
- **Matplotlib**
- **Jupyter Notebook**
- TIFF, SVS, CMG file formats

## Use Case

These tools were designed to automate image preprocessing for cancer cell segmentation and classification tasks, improving reproducibility and enabling more efficient training of deep learning models.

## Status

Scripts are functional and were used internally for preprocessing in a research setting. Further improvements (e.g., unit testing, CLI interfaces) are planned as future enhancements.

---

*Developed during a research term at BC Cancer, 2024.*
