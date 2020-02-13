import tifffile
from SEGMENTATION.segmentation import segmentation
from detection.deteccion import Particle, detect_particles, size_filter

import numpy as np


tif = tifffile.TiffFile('detection/Images_in/11.tif')
sequence = tif.asarray()

# sequence.shape[0]
for it in range (1):
	image = sequence[it,:,:]
	seg_img = segmentation(image)
	particles = detect_particles(image, seg_img)

print()


