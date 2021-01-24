# Dataset.py
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.cuda
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% 
class DatasetImage(Dataset):
    
    def __init__(self,dataframe,image_size,transform=None):
        
        # store dataframe with all paths for training/validation
        self.dataframe = dataframe
        
        # initiate transform
        self.transform = transform
        
        # define image size to normalize location 
        self.hieght = image_size[0]
        self.width = image_size[1]
        
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self,idx):
        
        # path to read the image
        img_path = self.dataframe['path'].iloc[idx]
        
        # read the image
        image = np.array(cv2.imread(img_path,0))
        
        # read coordinate
        x_tip = self.dataframe['X Coordinate'].iloc[idx]
        y_tip = self.dataframe['Y Coordinate'].iloc[idx]
        
        # normalize location from [0,1]
        x_tip = x_tip / self.hieght
        y_tip = y_tip / self.width
        
        # preform transforms
        if self.transform:
            image = self.transform(image).float()
        
        # create dictionary
        sample = {'image': image, 'label': torch.tensor([x_tip,y_tip]).float()}
        
        
        return sample
