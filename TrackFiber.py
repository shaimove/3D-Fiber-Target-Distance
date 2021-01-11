# TrackFiber.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ReadVideo

#%% define path
folder = '../2 Results from demo 6.12.20 with MotionTech/3D Tracking/'
video_left = folder + '100mm_left.mp4'
video_right = folder + '100mm_right.mp4'

frames_left,fps_left,size_left = ReadVideo.VideoCaptureData(video_left)
frames_right,fps_right,size_right = ReadVideo.VideoCaptureData(video_right)


#%% start tracking
# for the left video 
plt.close('all')
img0 = frames_left[0]
img0 = img0[220:320,580:680]

img1 = frames_left[1]

num_of_images = len(frames_left)-2


fiber_images = []
fiber_images.append(img0)

for i in tqdm(range(num_of_images)):
    # Perform ORB feature detection and description.
    orb = cv2.ORB_create()
    kp0, des0 = orb.detectAndCompute(img0, None)
    kp1, des1 = orb.detectAndCompute(img1, None)
    
    # Perform brute-force matching.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)
    
    # Sort the matches by distance.
    matches = sorted(matches, key=lambda x:x.distance)
    
    # Draw the best 25 matches.
    img_matches = cv2.drawMatches(
        img0, kp0, img1, kp1, matches[:3], img1,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # Show the matches.
    #plt.figure()
    #plt.imshow(img_matches)
    name_to_save = 'results/result' + str(i) + '.png'
    plt.imsave(name_to_save, img_matches)
    
    
    # take matches 
    fiber_keypt = kp0[matches[0].queryIdx].pt
    image_kept = kp1[matches[0].trainIdx].pt
    
    # now define the fiber image from new
    row = np.int(image_kept[1])
    col = np.int(image_kept[0])
    img0 = img1[row-50:row+50,col-50:col+50]
    fiber_images.append(img0)
    img1 = frames_left[i+2]
    
        
    
    
    
    
    
    