# TrackFiber.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils

#%% define path
folder = '../2 Results from demo 6.12.20 with MotionTech/3D Tracking/'
video_left = folder + '100mm_left.mp4'
video_right = folder + '100mm_right.mp4'

frames_left,fps_left,size_left = utils.VideoCaptureData(video_left)
frames_right,fps_right,size_right = utils.VideoCaptureData(video_right)


#%% Tracking loop
# define movies
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_writer = cv2.VideoWriter('results MeanShift track/frame.avi', fourcc, 30.0, (frames_left[0].shape[:2]))


for i in tqdm(range(len(frames_left)-1)):
    frame = frames_left[i]
    # Define an initial tracking window in the center of the frame.
    if i == 0:
        r = cv2.selectROI(frame)
        roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        track_window = (r)
        
        # calculate histogram and normalize the histogram
        mask = None
        roi_hist = cv2.calcHist([roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        # Define the termination criteria:
        # 10 iterations or convergence within 1-pixel radius.
        term_crit = (cv2.TERM_CRITERIA_COUNT , 10, 1)
        continue
    
    # take in grayscale image and claculate back projection
    back_proj = cv2.calcBackProject([frame], [0], roi_hist, [0, 180], 1)
    
    # Perform tracking with MeanShift.
    num_iters, track_window = cv2.meanShift(back_proj, track_window, term_crit)
    
    # Draw the tracking window.
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # save images
    frame_writer.write(frame)
    
    
    
frame_writer.release()   
 

    