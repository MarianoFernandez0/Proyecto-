import tifffile
from segmentation import segmentation
from skimage.io import imsave
import numpy as np


tif = tifffile.TiffFile('images/9.tif')
sequence = tif.asarray()

print(sequence.shape)
mask = segmentation(sequence[0,:,:])
imsave('images/sample_segmeneted1.jpg', mask)

print(np.amax(sequence))
#for image in (sequence.shape[0]):

