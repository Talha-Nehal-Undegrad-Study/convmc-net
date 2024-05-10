import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import glob
from skimage import io, color, img_as_float

from image_py_scripts import run_image_inpainting

import torch.utils.data as data
from pathlib import Path


ROOT = Path('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/convmc-net/Image_Inpainting_Data/BSDS300/images')
train_dir = ROOT / 'train'
test_dir = ROOT / 'test'

ground_truth_train_dir = train_dir / 'groundtruth'
lowrank_train_dir = train_dir / 'lowrank'

ground_truth_test_dir = test_dir / 'groundtruth'
lowrank_test_dir = test_dir / 'lowrank'

def make_imgs(split = 'train', shape = (150, 300), sampling_rate = 0.2, dB = 5.0):
    if split == 'train':
        jpg_files = list(train_dir.glob('*.jpg'))
        for idx, img in enumerate(jpg_files):
            image = io.imread(img)
            image = color.rgb2gray(img_as_float(image))
            image = np.resize(image, shape)

            np.save(os.path.join(ground_truth_train_dir, f'ground_image_MC_train_{idx}.npy'), image)
            
            image_lowrank = run_image_inpainting.add_gmm_noise(image = image, per = sampling_rate, dB = dB)
            np.save(os.path.join(lowrank_train_dir, f'lowrank_image_MC_train_{idx}.npy'), image_lowrank)
    
    else:
        jpg_files = list(test_dir.glob('*.jpg'))
        for idx, img in enumerate(jpg_files):
            image = io.imread(img)
            image = color.rgb2gray(img_as_float(image))
            image = np.resize(image, shape)

            np.save(os.path.join(ground_truth_test_dir, f'ground_image_MC_test_{idx}.npy'), image)
            
            image_lowrank = run_image_inpainting.add_gmm_noise(image = image, per = sampling_rate, dB = dB)
            np.save(os.path.join(lowrank_test_dir, f'lowrank_image_MC_test_{idx}.npy'), image_lowrank)

"""
Example Usage: make_imgs(split = 'train', shape = (150, 300), sampling_rate = 0.2, dB = 5.0)
"""