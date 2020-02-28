from skimage.io import imread, imsave
import skimage.color as color

import numpy as np

img = imread("Images_in/sample.jpg")
image_bw = color.rgb2gray(img)
grayscale = np.uint8(np.round(((image_bw - np.min(image_bw)) / (np.max(image_bw) - np.min(image_bw)) * 255)))

print(image_bw*255)
print(grayscale)
