import os
import statistics as s
import matplotlib.pyplot as plt

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

from dataloader import GetData
from loss import dice_metrics, precision_metrics, recall_metrics
import utils
from rgunet import RGUNet

def sort_files(items):
    keys = []
    sorted_files = []
    
    # Extract the file number 
    for file in items:
        num = os.path.basename(file).split('_')[1]
        slice_num = os.path.basename(file).split('_')[-1].split('.')[0]
        key = num + '_' + slice_num
        
        keys.append(key)
    
    # map the file number and the file
    scans = {keys[i]: items[i] for i in range(len(keys))}
    
    # sort the file number so it is is the order of slices
    sorted_list = sorted(keys,  key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    
    for key in sorted_list:
        sorted_files.append(scans[key])
        
    return sorted_files

def jpeg_2_nifti(array, path, n):
    # transform the array so it is in the correct orientation
    # array is the 3D array of concatenated 2D predicted masks
    # path is the path to output
    # n is just the number for file name
    
    mri = array
    mri = np.transpose(mri, (1, 2, 0))
    
    affine = np.eye(4)
    
    name = path + str(n) + '.nii.gz'
    converted_array = np.array(mri, dtype=np.float32)
    
    nifti_file = nib.Nifti1Image(converted_array, affine)
    nib.save(nifti_file, name)

""" Testing loop """
def testinge_loop (model, loader, path, device=torch.device('cuda')):
    n = 0
    i = 0
    mri = []
    
    with torch.no_grad():
        for x, y in loader:
            
            # split data to image and mask
            image = x
            mask = y
            image = image.to(device)
            mask = mask.to(device)
            
            # test_outputs = predicted mask
            test_outputs = model(image)
            test_outputs = torch.sigmoid(test_outputs)
            
            array = test_outputs[0].permute(1, 2, 0).detach().cpu().numpy()[:,:,0]
            if i ==154 :
                mri.append(array)
                
                jpeg_2_nifti(mri, path, n)
                mri = []
                i = 0
                n+=1
                
            else:
                i+=1
                mri.append(array)
                
if __name__ == '__main__':
    """ Seeding """
    utils.seeding(28)
    
    checkpoint_path = '' # change this 
    nifti_results = ''
    
    """ model """
    device = torch.device('cuda')
    model = RGUNet # load model
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    transform = A.Compose(
        [
            A.Normalize(mean=(0,0,0), 
                        std=(1,1,1),max_pixel_value=255.0),
            ToTensorV2()
            ])
    
    test_x = ['Dataset/T1/test/images/*',
              'Dataset/T1Gd/test/images/*',
              'Dataset/T2/test/images/*',
              'Dataset/Flair/test/images/*']
    
    test_y = ['Dataset/T1/test/labels/*',
              'Dataset/T1Gd/test/labels/*',
              'Dataset/T2/test/labels/*',
              'Dataset/Flair/test/labels/*']
    
    outputs = ['T1', 'T1Gd', 'T2', 'FLAIR']
    
    gts = ['data/T1/Test/labels/*',
          'data/T1Gd/Test/labels/*',
          'data/T2/Test/labels/*',
          'data/Flair/Test/labels/*']

    for i in range(len(test_x)):
        x = sort_files(glob(test_x[i]))
        y = sort_files(glob(test_y[i]))
        out_path = nifti_results + outputs[i] + '/'
        
        
        test_dataset = GetData(x, y, transform=transform) 
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
            )
        
        testinge_loop(model, test_loader, out_path)

