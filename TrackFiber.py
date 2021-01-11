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


#%% start tracking
# for the left video 
plt.close('all')
img0 = frames_left[0]
img0 = img0[220:320,580:680]

img = frames_right[0]

#%% create HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# find matching area in the image
found_rects, found_weights = hog.detectMultiScale(img,winStride=(4,4),
                                                  scale=1.02,finalThreshold=1.9)

#%% non-maximum suppression
found_rects_filtered = []
found_weights_filtered = []
    
for ri, r in enumerate(found_rects):
    # 
    for qi, q in enumerate(found_rects):
        #
        if ri != qi and utils.is_inside(r, q):
            break
        else:
            found_rects_filtered.append(r)
            found_weights_filtered.append(found_weights[ri])

#%% print detection
for ri, r in enumerate(found_rects_filtered):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % found_weights_filtered[ri]
    cv2.putText(img, text, (x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
cv2.imshow('Women in Hayfield Detected', img)
cv2.imwrite('./women_in_hayfield_detected.jpg', img)
cv2.waitKey(0)    
    
    
    
    
    
    
    
    
    
    
    