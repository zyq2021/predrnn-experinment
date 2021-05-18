from skimage.measure import compare_ssim
from scipy.misc import imread
import numpy as np
 
img1 = imread('C:\\Users\\40583\\Desktop\\2020-12-24 19_09_08.jpg')
img2 = imread('C:\\Users\\40583\\Desktop\\2020-12-24 19_09_13.jpg')
 
img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
 
print(img2.shape)
print(img1.shape)
ssim = compare_ssim(img1, img2, multichannel=True)
 
print(ssim)
