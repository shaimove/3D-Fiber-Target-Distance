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

#%% Going over list of points from ORB and return the closet point in image

def Find_Nearest_Point(point_to_detect,list_of_points,min_distance=1000000):
    
    # Define loop variabels
    nearest_point = None
    nearest_distance = min_distance
    index_nearest_point = None
    nearest_keypoint = None
    
    # Loop
    for i,point in enumerate(list_of_points):
        # calculate L2
        distance = ((point_to_detect[0] - point.pt[0])**2 + 
                    (point_to_detect[1] - point.pt[1])**2)**0.5
        
        # if we found shorter distance, update!
        if distance < nearest_distance:
            nearest_point = int(point.pt[0]),int(point.pt[1]) # update (x,y)
            index_nearest_point = i # update index
            nearest_keypoint = list_of_points[i] # update keypoint
            nearest_distance = distance
    
    return nearest_point,nearest_keypoint,index_nearest_point,nearest_distance
    


#%% Create keypoint list from macther

def Create_Keypoints_from_Matcher(matches,keypoint_lists):
    # first, sort them by distance from descriptor
    #matches = sorted(matches, key=lambda x:x.distance)
    
    # list
    new_keypoint_list = []
    
    for match in matches:
        ind = match.trainIdx
        new_keypoint_list.append(keypoint_lists[ind])
    
    return new_keypoint_list






