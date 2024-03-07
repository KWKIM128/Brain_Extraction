""" 
Apply post-processing before evaluating the predicted masks.
"""
from glob import glob
from loss import dice_metrics, precision_metrics, recall_metrics
import numpy as np

from rgunetv5 import ARGUNet
from lighterunet import LighterUnet
import os
from rgunet import RGUNet
from gunet import GUNet


import nibabel as nib

import statistics as s

if __name__ == '__main__':

    nifti_results = 'post_processed/Unet/'
    
    outputs = ['T1', 'T1Gd', 'T2', 'FLAIR']
    
    gts = ['data/T1/Test/labels/*',
          'data/T1Gd/Test/labels/*',
          'data/T2/Test/labels/*',
          'data/Flair/Test/labels/*']

    for i in range(len(outputs)):
        out_path = nifti_results + outputs[i] + '/'
        
        predicted = sorted(glob(out_path + '*.nii.gz'))
        predicted.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))
        ground_truth = sorted(glob(gts[i]))
        
        dices = []
        precisions = []
        recalls = []
        
        for j in range(len(predicted)):
            pm = nib.load(predicted[j]).get_fdata()
            gt = nib.load(ground_truth[j]).get_fdata()
            
            pm = pm > 0.5
            gt = gt > 0.5
            
            
            dice = dice_metrics(pm, gt)
            precision = precision_metrics(pm, gt)
            recall = recall_metrics(pm, gt)
            
            dices.append(dice)
            precisions.append(precision)
            recalls.append(recall)
        
        print(outputs[i])
        print(s.median(dices))
        print(s.stdev(dices))
