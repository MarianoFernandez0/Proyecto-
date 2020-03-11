import tifffile
import pandas as pd
from SEGMENTATION.segmentation import segmentation
from detection.deteccion import detect_particles, size_filter
from fluorescence.fluorescence import fluorescence

import skimage.color as color
import numpy as np

tif = tifffile.TiffFile('detection/Images_in/9.tif')
sequence = tif.asarray()

# crear dataframe
data = pd.DataFrame(columns=['x', 'y', 'frame', 'ctcf', 'mean_gray_value'])

for nro_frame in range(1):
    image = sequence[nro_frame, :, :]
    seg_img = segmentation(image)
    particles = detect_particles(seg_img)

    image_bw = color.rgb2gray(image)
    grayscale = np.uint8(np.round(((image_bw - np.min(image_bw)) / (np.max(image_bw) - np.min(image_bw)) * 255)))

    for index, row in particles.iterrows():
        # fluorescencia
        mask = row['mask']
        ctcf, mean_gray_value = fluorescence(grayscale, mask, seg_img / 255)
        # rellenar dataframe
        data = data.append({'x': row['x'], 'y': row['y'], 'ctcf': ctcf, 'mean_gray_value': mean_gray_value},
                           ignore_index=True)

    data['frame'] = nro_frame

# print(data)
# print(particles)
print(sequence.shape)
