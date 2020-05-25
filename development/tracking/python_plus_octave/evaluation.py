from segmentation import segmentation
from detection import detect_particles, size_filter
from fluorescence import fluorescence

import pandas as pd
import numpy as np
import skimage.color as color


def evaluation(tif, include_mask=False):
    '''
    Entrada: .tif
            include_mask (boolean): Determina si el dataframe de salida incluye a la máscara de cada partícula detectada.
    Salida: dataframe

    Funcion que dado un .tif devuelve un dataframe con los campos {x, y, frame, ctcf, mean_gray_value}.
    En el dataframe se guardan los resultados de los algoritmos implementados: segmentation, detect_particles y fluorescence
    En las columnas x, y se guardan las posiciones de las particulas y en frame el valor correspondiente del frame en el cual
     aparecen la particula, luego de aplicar la función segmentation y detect_particles.
    En las columnas ctcf y mean_gray_value se guardan los valores correspondientes devueltos por la función fluorescence.
    '''
    sequence = tif.asarray()
    data = pd.DataFrame(columns=['x', 'y', 'frame', 'ctcf', 'mean_gray_value'])

    for nro_frame in range(sequence.shape[0]):
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
            if include_mask:
                data = data.append(
                    {'x': row['x'], 'y': row['y'], 'frame': nro_frame, 'ctcf': ctcf,
                     'mean_gray_value': mean_gray_value, 'mask': row['mask']}, ignore_index=True)
            else:
                data = data.append(
                    {'x': row['x'], 'y': row['y'], 'frame': nro_frame, 'ctcf': ctcf,
                     'mean_gray_value': mean_gray_value}, ignore_index=True)

    return data
