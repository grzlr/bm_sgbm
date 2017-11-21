import numpy as np
import cv2
from matplotlib import pyplot as plt

print('loading images...')
imgL = cv2.imread('../data/left.png', 0)
imgR = cv2.imread('../data/right.png', 0)

# SAD window size should be between 5..255
block_size = 15

min_disp = 0
num_disp = 320 - min_disp
uniquenessRatio = 10


stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setUniquenessRatio(uniquenessRatio)


# disparity = stereo.compute(imgL,imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print "saving disparity as disparity_image_bm.txt"
np.savetxt("../data/disparity_image_bm.txt", disparity, fmt='%3.2f', delimiter=' ', newline='\n')


print "plotting disparity"
plt.imshow(disparity,'gray')
plt.show()
