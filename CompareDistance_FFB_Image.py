# CompareDistance.py
# create graph comparing the distance as measured the the FFB system and the 
# output of the TEMA software
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

#%% Read files
camera = '../2 Results from demo 6.12.20 with MotionTech/Cameras.xlsx'
FFB = '../2 Results from demo 6.12.20 with MotionTech/FFB.xlsx'

camera_file = pd.read_excel(camera)
FFB_file = pd.read_excel(FFB)

#%% Analysis 
dist_camera = camera_file.iloc[2:,1].to_numpy()
dist_ffb = FFB_file.iloc[:,-2].to_numpy()

# ffb is longer than camera, so we need to sample and sync the time 
len_camera = len(dist_camera)
len_ffb = len(dist_ffb)

ind_to_sample = np.int64(np.linspace(1700,len_ffb-800,len_camera))
dist_ffb_sampled = dist_ffb[ind_to_sample]


plt.figure()
plt.plot(dist_camera,label='camera')
plt.plot(dist_ffb_sampled,label='FFB')
plt.grid(); plt.legend(); plt.title('Compare measurement of 2D system with FFB')
plt.xlabel('Time [a.u]'); plt.ylabel('Distance [mm]')
