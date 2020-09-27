import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# import skimage.color as color
from src.detection.segmentation import segmentation
from src.fluorescence.fluorescence import fluorescence
from src.detection.detection import detect_particles, size_filter


def evaluation(sequence, pixel_size, include_mask=False):
    """
        Funcion que dado un .tif devuelve un dataframe con los campos {x, y, frame, ctcf, mean_gray_value}.
        En el dataframe se guardan los resultados de los algoritmos implementados: segmentation, detect_particles
        y fluorescence.
        En las columnas x, y se guardan las posiciones de las particulas y en frame el valor correspondiente del frame
        en el cual aparecen la particula, luego de aplicar la función segmentation y detect_particles.
        En las columnas ctcf y mean_gray_value se guardan los valores correspondientes devueltos por la función
        fluorescence.
    Args:
        sequence (np.ndarray): video.
        include_mask (boolean): Determina si el dataframe de salida incluye a la máscara de cada partícula detectada.
    Returns:
        data (pd.DataFrame): Partículas detectadas en cada frame del video.
    """
    # sequence = tif.asarray()
    data = pd.DataFrame(columns=['x', 'y', 'frame', 'ctcf', 'mean_gray_value'])

    for nro_frame in tqdm(range(sequence.shape[0])):
        image = sequence[nro_frame, :, :]
        seg_img = segmentation(image)
        particles = detect_particles(seg_img)
        particles = size_filter(particles, pixel_size=[pixel_size, pixel_size])

        image_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image_bw = color.rgb2gray(image)
        grayscale = np.uint8(np.round(((image_bw - np.min(image_bw)) / (np.max(image_bw) - np.min(image_bw)) * 255)))
        for index, row in particles.iterrows():
            # fluorescencia
            mask = row['mask']
            ctcf, mean_gray_value = fluorescence(mask, grayscale, seg_img / 255)
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
