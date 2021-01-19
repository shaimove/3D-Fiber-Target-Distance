# CreateImageDataset.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import utils

#%% Step 1: Read video and define output video
folder = '../2 Results from demo 6.12.20 with MotionTech/3D Tracking/'
video_left = folder + '100mm_left.mp4'
video_right = folder + '100mm_right.mp4'

# read video files
frames_left,fps_left,size_left = utils.VideoCaptureData(video_left)
frames_right,fps_right,size_right = utils.VideoCaptureData(video_right)

# read tables
table_name = video_left[:-4] + '.csv'
table_left = pd.read_csv(table_name)

table_name = video_right[:-4] + '.csv'
table_right = pd.read_csv(table_name)

# number of images to read
num_left = len(table_left)
num_right = len(table_right)

#%% Now save in different folder, all images from two video file, and
# create new unified dataframe
folder_to_save = '../Dataset/'
new_dataframe = pd.DataFrame()

# 5 columns in the dataset
x_point = []; y_point = []; video_source = []; path = []; frame_in_video = []
mean_pixel_value = []

for i in tqdm(range(num_left)):
    # convert to grayscale and save
    frame = frames_left[i]
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image_name = 'left_' + str(i) + '.jpg'
    path_to_save = folder_to_save + image_name
    cv2.imwrite(path_to_save,image)
    
    # read data from dataframe to create united dataframe
    path.append(path_to_save)
    x_point.append(table_left['X Coordinate'].iloc[i])
    y_point.append(table_left['Y Coordinate'].iloc[i])
    video_source.append('left')
    frame_in_video.append(i)
    mean_pixel_value.append(np.mean(image))


for i in tqdm(range(num_right)):
    # convert to grayscale and save
    frame = frames_right[i]
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image_name = 'right_' + str(i) + '.jpg'
    path_to_save = folder_to_save + image_name
    cv2.imwrite(path_to_save,image)
    
    # read data from dataframe to create united dataframe
    path.append(path_to_save)
    x_point.append(table_right['X Coordinate'].iloc[i])
    y_point.append(table_right['Y Coordinate'].iloc[i])
    video_source.append('right')
    frame_in_video.append(i)
    mean_pixel_value.append(np.mean(image))

# save dataframe
new_dataframe['path'] = path
new_dataframe['video_source'] = video_source
new_dataframe['frame_in_video'] = frame_in_video
new_dataframe['X Coordinate'] = x_point
new_dataframe['Y Coordinate'] = y_point
new_dataframe['Mean Pixel Value'] = mean_pixel_value

filename = folder_to_save + 'Dataset.csv'
new_dataframe.to_csv(filename)
