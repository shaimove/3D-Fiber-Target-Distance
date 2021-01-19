# ORB_Basic.py
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

frames_left,fps_left,size_left = utils.VideoCaptureData(video_left)
frames_right,fps_right,size_right = utils.VideoCaptureData(video_right)

# Define video to save
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_writer = cv2.VideoWriter('right tip of fiber.avi', fourcc, 10.0, (frames_left[0].shape[:2]))


#%% Step 2: Create DataFrame with all paths
table_right = pd.DataFrame() # path to video file, frame number, x_point, y_point

num_left = len(frames_left); num_right = len(frames_right)
x_points = []
y_points = []
frame_num = []

#%% Loop left frame

for i,frame in tqdm(enumerate(frames_right[:-60])):
    # mark tip of the fiber
    plt.figure(figsize=(10,10))
    plt.imshow(frame)
    pts = plt.ginput(1)
    plt.close('all')
    
    # append to list of points
    x_points.append(int(pts[0][0]))
    y_points.append(int(pts[0][1]))
    frame_num.append(i)
    
    # mark point at frame and save to new video
    cv2.circle(frame, (int(pts[0][0]),int(pts[0][1])),
               radius=5,color=(0, 0, 255),thickness=2)
    frame_writer.write(frame)
    
    
#%% Assign values
# Fix
point_to_erase_from = 442
x_points[point_to_erase_from:] = []
y_points[point_to_erase_from:] = []
frame_num[point_to_erase_from:] = []

#%%
table_right['Path'] = video_right
table_right['Frame Number'] = frame_num
table_right['X Coordinate'] = x_points
table_right['Y Coordinate'] = y_points

# save the table to CSV file  
filename = video_right[:-4] + '.csv'
table_right.to_csv(filename)

# save movie
frame_writer.release()   









