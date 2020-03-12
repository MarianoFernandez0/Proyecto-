import numpy as np
from skimage.io import imread
import cv2
from fluorescence import fluorescence

original_img = imread('Images_in/sample.jpg')
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
mask = imread('Images_in/mascara.png') / 255
segmented_img = imread('Images_in/sample_segmented.jpg') / 255

ctcf, mean_gray_value = fluorescence(gray_img, mask, segmented_img)
print(ctcf, mean_gray_value)
