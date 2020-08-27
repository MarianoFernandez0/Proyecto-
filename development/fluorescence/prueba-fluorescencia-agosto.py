from skimage.io import imread, imsave
from segmentation import segmentation
from deteccion import detect_particles
from fluorescence import fluorescence

import pandas as pd
import numpy as np
import skimage.color as color

img_prueba = imread("Images_in/sample-recorte.png") #CARGAR IMG

segmented_img = segmentation(img_prueba) #SEGMENTACION
particles = detect_particles(segmented_img) #DETECCION

### PASAR img_prueba A GRISES ###
image_bw = color.rgb2gray(img_prueba)
grayscale = np.uint8(np.round(((image_bw - np.min(image_bw)) / (np.max(image_bw) - np.min(image_bw)) * 255)))

data = pd.DataFrame(columns=['x', 'y', 'frame', 'ctcf', 'mean_gray_value'])
for index, row in particles.iterrows():
    mask = row['mask']
    ctcf, mean_gray_value = fluorescence(grayscale, mask, segmented_img / 255)
    #print('funcion fluo', index, mean_gray_value)
    # rellenar dataframe
    data = data.append({'x': row['x'], 'y': row['y'], 'frame': '0', 'ctcf': ctcf,
             'mean_gray_value': mean_gray_value, 'mask': row['mask']}, ignore_index=True)

    ### GUARDAR LAS MASCARAS EN BLANCO Y NEGRO ###
    #mask_bw = color.rgb2gray(mask)
    #mask_grayscale = np.uint8(np.round(((mask_bw - np.min(mask_bw)) / (np.max(mask_bw) - np.min(mask_bw)) * 255)))
    #imsave('Images_out/mascara%d.png' %index, mask_grayscale)
    ##############################################

    ###############
    fluorescent_mask = grayscale * mask
    imsave('Images_out/fluorescent_mask%d.png' %index , np.uint8(fluorescent_mask))
    integrated_density = np.sum(fluorescent_mask)
    area_in_pixels = np.sum(mask)
    mean_gray_value = integrated_density / area_in_pixels
    print(index, integrated_density, area_in_pixels, mean_gray_value)

data.to_csv('data.csv', index=False, header=True)
imsave('Images_out/sample_gris.png',grayscale)
