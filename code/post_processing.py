import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from scipy import ndimage as ndi
 
def post_processing(mask):
    """
    The following function is an adaptation of a function developed by Lucena 
    et al. 2019. Convolutional neural networks for skull-stripping in brain MR 
    imaging using silver standard masks. Artificial intelligence in medicine
    98, pp. 48–58.
    
    Original code: https://github.com/MICLab-Unicamp/CONSNet/blob/master/infer.py
    """
  
    # Threshold to get rid of noise
    mask = np.where(mask >= 0.5, 1, 0)
    labels, n = ndi.measurements.label(mask)
    hist = np.histogram(labels.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    i = np.argmax(hist[1:]) + 1
    mask = (mask != i).astype(np.uint8)
    mask, n = ndi.measurements.label(mask)
    hist = np.histogram(mask.flat, bins=(n + 1), range=(-0.5, n + 0.5))[0]
    i = np.argmax(hist[1:]) + 1
    return (mask != i).astype(np.uint8)
 
def applying_mask (original_scan, mask, output_path):
  
    file_name = os.path.basename(original_scan)
    outputfile_name = os.path.splitext(file_name)[0]
    outputfile_name = outputfile_name.split('.')[0]+ '_brainextracted.nii.gz'
    output = os.path.join(output_path, outputfile_name)
    original_mri = nib.load(original_scan)
    original_data = original_mri.get_fdata()
    mask = nib.load(mask).get_fdata()
 
    # Apply the mask to the original MRI data
    masked_data = original_data * mask
 
 
    # Create a new NIfTI image with the masked data
    masked_mri = nib.Nifti1Image(masked_data, original_mri.affine)
    nib.save(masked_mri, output)
