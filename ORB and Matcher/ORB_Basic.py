# ORB_Basic.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
frame_writer = cv2.VideoWriter('frame.avi', fourcc, 10.0, (frames_left[0].shape[:2]))


#%% Step 2: Mark the tip of the fiber and it's ORB descriptor
# Mark the tip of the fiber
first_image = frames_left[0]

plt.imshow(first_image)
pts = plt.ginput(1)
point = int(pts[0][0]),int(pts[0][1])
plt.close('all')

#%% Step 3: Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB/SIFT for both first image and image with tip of the fiber
kp0, des0 = orb.detectAndCompute(first_image, None)

# find closest point
n_point,n_index,dist = utils.Find_Nearest_Point(point,kp0)
refference_descriptor = np.expand_dims(des0[n_index], axis=0)


#%% Step 4: Initiate Loop
# Define lists
tip_fiber_detected = []
descriptors = []
distances = []

# append to list
tip_fiber_detected.append(n_point)
descriptors.append(refference_descriptor)
distances.append(dist)

previous_tip_fiber_detected = n_point

# Define brute-force matching.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Define FLANN parameters
# ORB parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary


flann = cv2.FlannBasedMatcher(index_params,search_params)

#%% Loop
for i,frame in tqdm(enumerate(frames_left[0:-1])):
    # crop the image using the point given from previous iteration, to limit area search
    image_crop,crop_points = utils.Crop_Image(
        frame,previous_tip_fiber_detected,window_size=200)
    
    # keypoints and descriptors from new image
    #kp1, des1 = orb.detectAndCompute(frame, None)
    kp1, des1 = orb.detectAndCompute(image_crop, None)
    
    # Perform brute-force/FLANN matching
    matches = bf.match(refference_descriptor, des1)
    #matches = flann.knnMatch(refference_descriptor,des1,k=2)[0]

    # Create list of keypoints from matcher 
    new_keypoint_list = utils.Create_Keypoints_from_Matcher(matches,kp1)
    
    # find the closest point from matcher to previous tip_fiber_detected
    n_point,n_index,dist = utils.Find_Nearest_Point(
        previous_tip_fiber_detected,new_keypoint_list,crop_points)
    
    # take descriptor of new nearest_point
    descriptor_detected = np.expand_dims(des1[n_index],axis=0)
    
    # append new values to lists
    tip_fiber_detected.append(n_point); 
    descriptors.append(descriptor_detected); distances.append(dist)
        
    # define descriptor for next iteration
    refference_descriptor = descriptor_detected
    previous_tip_fiber_detected = n_point

        
    # plot the tip of fiber and save to a movie
    img = frame.copy()
    cv2.circle(img, (int(n_point[0]),int(n_point[1])), radius=5,color=(0, 0, 255),thickness=2)
    frame_writer.write(img)
    




frame_writer.release()   


#%% Plot distance and travel
travel_x,travel_y = np.array(tip_fiber_detected)[:,0],np.array(tip_fiber_detected)[:,0]
distances = np.array(distances)

plt.figure()
plt.scatter(travel_x,travel_y); plt.title('Position in time')

plt.figure()
plt.plot(range(distances.shape[0]),distances); plt.title('Distance between consecutive frames')
plt.xlabel('Time [a.u]'); plt.ylabel('Distance [a.u]'); plt.grid()












