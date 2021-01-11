# ReadVideo.py
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    
    
    return frames,fps,size


#%% find if one rectangle inside the other

def is_inside(i, o):
    # inner rectangle
    ix, iy, iw, ih = i
    # outter rectangle
    ox, oy, ow, oh = o
    
    return ix > ox and ix + iw < ox + ow and \
        iy > oy and iy + ih < oy + oh






