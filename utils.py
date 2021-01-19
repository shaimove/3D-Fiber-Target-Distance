# ReadVideo.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


#%% function to read video files
def VideoCaptureData(video_path):
    # get data about the video
    VideoCapture  = cv2.VideoCapture(video_path) 
    fps = VideoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # get the frames
    frames = []
    success = True
    
    while success:
        success, frame = VideoCapture.read()
        frames.append(frame)
    
    
    return frames,fps,size


#%% Calculate stats
def CalculateStats(data_train,image_size):

    # read DataFrame
    dataframe = data_train
    
    # calculate mean based on previous calculation
    mean = np.mean(np.array(dataframe['Mean Pixel Value']))
    
    # define paths
    images_path = dataframe['path']
    
    # accumalated sum and number of images
    acc_std = 0
    num_of_pixels = len(images_path) * image_size[0] * image_size[1]

    for path in tqdm(images_path):
        # read image
        image = np.array(cv2.imread(path,0))
        
        # subtract the mean and square and sum
        image_mse = np.sum(np.square(image - mean))
        acc_std += image_mse
        
    # divide by (n-1) and sqrt
    std = np.sqrt(acc_std / (num_of_pixels-1))
    
    return mean,std

#%% Count number of parameters in model
def count_parameters(model):
    num_parmas = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Parmaters of this Model is: %s parameters' % num_parmas)
    return None


    