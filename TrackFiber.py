# TrackFiber.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from FiberClass import Fiber

#%% define path
folder = '../2 Results from demo 6.12.20 with MotionTech/3D Tracking/'
video_left = folder + '100mm_left.mp4'
video_right = folder + '100mm_right.mp4'

frames_left,fps_left,size_left = utils.VideoCaptureData(video_left)
frames_right,fps_right,size_right = utils.VideoCaptureData(video_right)


#%% Tracking loop
# define movies
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_writer = cv2.VideoWriter('results Kalman filter track/frame.avi', fourcc, 30.0, (frames_left[0].shape[:2]))


for i in tqdm(range(len(frames_left)-1)):
    frame = frames_left[i]
    # Iinitialize Fiber detection
    if i == 0:
        # mark rectangle
        r = cv2.selectROI(frame)
        roi = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        track_window = (r)
        cv2.destroyAllWindows()
        
        # mark point
        plt.imshow(frame)
        pts = plt.ginput(3)
        point = int(pts[0][0]),int(pts[0][1])
        plt.close('all')
        
        # initialize Fiber 
        fiber = Fiber(0,frame,track_window,point)
        
    # Update the tracking of the fiber.
    fiber.update(frame)

    # save images
    frame_writer.write(frame)
    
    
    
frame_writer.release()   
 
