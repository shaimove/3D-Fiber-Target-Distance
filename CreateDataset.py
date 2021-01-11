# CreateDataset.py
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

#%% choose randomly, image to create from them dataset
num_of_img = len(frames_left)
frames = np.linspace(0,num_of_img-1,num_of_img)
frames = np.random.choice(frames,50)
frames = np.int64(frames)


#%% create negative dataset

for frame in frames:
    # choose frame
    img = frames_left[frame]
    
    # crop positive image
    r = cv2.selectROI("Select the fiber",img)
    fiber = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    fiber_path = 'Positive/pos-' + str(frame) + '.jpg'
    cv2.imwrite(fiber_path,fiber)
    cv2.destroyAllWindows()
    
    # crop positive image
    r = cv2.selectROI("Select the non-fiber",img)
    nonfiber = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    fiber_path = 'Negative/neg-' + str(frame) + '.jpg'
    cv2.imwrite(fiber_path,nonfiber)
    cv2.destroyAllWindows()


