import numpy as np


def fluorescence(gray_img, mask, segmented_img):
    '''
    Indica los valores CTCF (Corrected Total Cell Fluorescence) y mean gray value (promedio del valor de gris) para una mascara dada, los cuales indican dos medidas de fluorescencia para las particulas detectadas
    Entradas:
        mask: imagen de tamaño NxM la cual muestra el area perteneciente a una particula con valor 1 y el resto 0 (atributo de la clase particle)
        gray_img: imagen original de tamaño NxM en valores de grises
        segmented_img: imagen segmentada de tamaño NxM (salida de la función segmentation)
    Salida:
        CTCF y mean_gray_value - medidas de fluorescencia para la mascara dada
    '''
    # contruccion imagen "fluorescent mask"
    fluorescent_mask = gray_img * mask
    integrated_density = np.sum(fluorescent_mask)
    area_in_pixels = np.sum(mask)
    mean_gray_value = integrated_density / area_in_pixels

    # construccion imagen "background"
    segmented_img_inv = (segmented_img == 0).astype(np.uint8)
    background_img = gray_img * segmented_img_inv
    background_mean = np.sum(background_img) / (gray_img.shape[0] * gray_img.shape[1])
    CTCF = integrated_density - (area_in_pixels * background_mean)

    return CTCF, mean_gray_value
