#!/home/leo/anaconda3/bin/python
import cv2
import numpy as np


def segmentation(image):
    '''
    Segmenta una imagen dada.
    En el estudio para el proyecto, se definió
    que filtrar la imagen y luego segmentar utilizando OTSU
    fue el mejor desempeño
    Entradas:
        image: imagen de tamaño NxMx3 la cual se desea segmentar
    Salida:
        Máscara de tamaño NxMx1 con la imagen segmentada
    '''
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_tresh = cv2.threshold(image_bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_streched = cv2.morphologyEx(image_tresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(image_streched, np.ones((3, 3), np.uint8), iterations=1)

    return mask
