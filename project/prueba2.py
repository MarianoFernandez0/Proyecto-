import tifffile
import pandas as pd
from SEGMENTATION.segmentation import segmentation
from detection.deteccion import detect_particles, size_filter

import numpy as np


tif = tifffile.TiffFile('detection/Images_in/11.tif')
sequence = tif.asarray()

for nro_frame in range (1):
	image = sequence[nro_frame,:,:]
	seg_img = segmentation(image)
	particles = detect_particles(seg_img)

print(particles)