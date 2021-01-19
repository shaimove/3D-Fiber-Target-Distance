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

def Find_Nearest_Point(point_to_detect,list_of_points,crop_points=None):
    
    # Define loop variabels
    nearest_point = None
    nearest_distance = 10000000
    index_nearest_point = None
    nearest_keypoint = None
    
    # create new list from points in list_of_points
    list_of_points_new = []
    
    for point in list_of_points:
        x = point.pt[0]; y = point.pt[1]
        
        # if crop_points is None, don't fix points.
        if crop_points is not None:
            x = x + crop_points[0]
            y = y + crop_points[2]
            list_of_points_new.append((x,y))
        else:
            list_of_points_new.append((x,y))
    
    # Loop to find shortest distance
    for i,point in enumerate(list_of_points_new):
        # calculate L2
        distance = ((point_to_detect[0] - point[0])**2 + 
                    (point_to_detect[1] - point[1])**2)**0.5
        
        # if we found shorter distance, update!
        if distance < nearest_distance:
            nearest_point = point[0],point[1] # update (x,y)
            index_nearest_point = i # update index
            nearest_distance = distance
    
     
    
    return nearest_point,index_nearest_point,nearest_distance
    


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


#%% Create a cropped image of minimum size 100*100 to search in limited area

def Crop_Image(image,point,window_size=150):
    # taking size
    rows,cols,rgb = image.shape
    
    # take x and y points and convert to integers
    point_x,point_y = point
    point_x = int(point_x); point_y = int(point_y)
    
    # define h_min, h_max
    if (point_x + window_size/2 > rows):
        h_min = rows - int(window_size)
        h_max = rows
    elif (point_x - window_size/2 < 0):
        h_min = 0
        h_max = int(window_size)
    else:
        h_min = point_x - int(window_size/2)
        h_max = point_x + int(window_size/2)
    
    
    # define w_min and w_max
    if (point_y + window_size/2 > cols):
        w_min = cols - int(window_size)
        w_max = cols
    elif (point_y - window_size/2 < 0):
        w_min = 0
        w_max = 0
    else:
        w_min = point_y - int(window_size/2)
        w_max = point_y + int(window_size/2)
    
    # cropping
    image_crop = image[h_min:h_max,w_min:w_max,:]
    crop_points = [h_min,h_max,w_min,w_max]
    
    return image_crop,crop_points
    
    
    
    



