#!/home/leo/anaconda3/bin/python
import skimage.data as data
import skimage.filters as filters
import skimage.color as color
import scipy.ndimage as nd


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
    image_bw = color.rgb2gray(image)
    image_filtered = nd.gaussian_filter(image_bw, sigma=3)
    image_threshold = filters.threshold_otsu(image_filtered)
    mask = image_bw > image_threshold
    return mask
