#!/home/leo/anaconda3/bin/python
from skimage.io import imread
from skimage.io import imsave
from segmentation import segmentation

image = imread('IMAGENES/sample.jpg')
mask = segmentation(image)
imsave('IMAGENES/sample_segmentd.jpg', mask)