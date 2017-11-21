#!/usr/bin/env python

# skeleton source https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
import numpy as np
import cv2
from matplotlib import pyplot as plt

print('loading images...')
imgL = cv2.imread('../data/left.png', 0)  
imgR = cv2.imread('../data/right.png', 0) 

# disparity range tuning
# https://docs.opencv.org/trunk/d2/d85/classcv_1_1StereoSGBM.html
window_size = 3
min_disp = 0
num_disp = 320 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 3,
    P1 = 8 * 3 * window_size**2,
    P2 = 32 * 3 * window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
    )

print('computing disparity...')
# disparity = stereo.compute(imgL, imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print "saving disparity as disparity_image_sgbm.txt"
np.savetxt("../data/disparity_image_sgbm.txt", disparity, fmt = '%3.2f', delimiter = ' ', newline = '\n')

# plt.imshow(imgL, 'gray')
plt.imshow(disparity, 'gray')
# plt.imshow('disparity', (disparity - min_disp) / num_disp)
plt.show()


