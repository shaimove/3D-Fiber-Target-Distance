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
# erode, and dilate kernels
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# define background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# define movies
fourcc = cv2.VideoWriter_fourcc(*'XVID')
mog_writer = cv2.VideoWriter('results MOG track/mog.avi', fourcc, 30.0, (frames_left[0].shape[:2]))
frame_writer = cv2.VideoWriter('results MOG track/frame.avi', fourcc, 30.0, (frames_left[0].shape[:2]))


for i in tqdm(range(len(frames_left)-1)):
    frame = frames_left[i]
    # background subtractor and threshold the image
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    
    # binary operations
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    
    # find and plot changes with rectangles
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    # save images
    mog_writer.write(cv2.cvtColor(fg_mask*255,cv2.COLOR_GRAY2BGR))
    frame_writer.write(frame)
    
    
    
mog_writer.release()   
frame_writer.release()   
 

    