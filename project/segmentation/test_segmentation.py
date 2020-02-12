#!/home/leo/anaconda3/bin/python
from skimage.io import imread
from skimage.io import imsave
from segmentation import segmentation

image = imread('images/sample.jpg')
mask = segmentation(image)
imsave('images/sample_segmeneted.jpg', mask)