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
# blur, erode, and dilate kernels
BLUR_RADIUS = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

# define movies
fourcc = cv2.VideoWriter_fourcc(*'XVID')
diff_writer = cv2.VideoWriter('results basic track/diff.avi', fourcc, 30.0, (frames_left[0].shape[:2]))
thresh_writer = cv2.VideoWriter('results basic track/thresh.avi', fourcc, 30.0, (frames_left[0].shape[:2]))
frame_writer = cv2.VideoWriter('results basic track/frame.avi', fourcc, 30.0, (frames_left[0].shape[:2]))


for i in tqdm(range(len(frames_left)-1)):
    frame = frames_left[i]
    # take refference image
    if i == 0:
        # grayscale and blur
        gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.GaussianBlur(gray_background,(BLUR_RADIUS, BLUR_RADIUS), 0)
        continue
    
    # take the current image
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(BLUR_RADIUS, BLUR_RADIUS), 0)
    
    # find difference between current image to refference image and threshold
    diff = cv2.absdiff(gray_background,gray_frame)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    
    # binary operations
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    
    # find and plot changes with rectangles
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 4000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    # save images
    diff_writer.write(cv2.cvtColor(diff*255,cv2.COLOR_GRAY2BGR))
    thresh_writer.write(cv2.cvtColor(thresh*255,cv2.COLOR_GRAY2BGR))
    frame_writer.write(frame)
    
    
    
diff_writer.release()   
thresh_writer.release()  
frame_writer.release()   
 

    