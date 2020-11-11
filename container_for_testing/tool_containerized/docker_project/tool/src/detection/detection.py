import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label


def detect_particles(seg_img):
	"""
	Toma la imagen original y la segmentada como entrada, devuelve un dataframe con todas las partículas
	de la imagen y sus propidades.

	Args:
		seg_img (np.ndarray): imagen segmentada, dim (M,N).

	Returns:
		particles (pd.DataFrame): Dataframe con todas las partículas, cols (id, x, y, total_pixels, mask).
	"""

	M = seg_img.shape[0]
	N = seg_img.shape[1]

	# Etiqueta cada partícula con un entero diferente
	labeled_img, total_particles = label(seg_img, connectivity=2, return_num=True)

	count = 0
	particles = pd.DataFrame(index=range(total_particles), columns=['id', 'x', 'y', 'total_pixels', 'mask'])

	# Se recorren todos los pixeles de la imágen para encontrar el centro geométrico de cada partícula haciendo
	# el promedio de sus coordenadas además se guardan el resto de las propiedades de las partículas
	for p in range(1, total_particles+1):
		particles.loc[p - 1, ['id']] = p
		coords_p = np.argwhere(labeled_img == p)
		particles.loc[p-1, ['x']] = np.mean(coords_p[:, 0])
		particles.loc[p-1, ['y']] = np.mean(coords_p[:, 1])
		particles.loc[p-1, ['total_pixels']] = coords_p.shape[0]
		mask = np.zeros((M, N))
		mask[labeled_img == p] = 1
		particles.loc[p-1, ['mask']] = [mask]

	return particles


def size_filter(particles, pixel_size):
	"""
	Toma la lista de partículas y filtra las que son menores a 10 micrometros cuadrados.

	Parameters:
		particles (df(id, coord_x, coord_y, total_pixels, mask)): DataFrame de partículas a filtrar.
		pixel_size (list(float,float)): Las dimensiones de un pixel en micrometros.

	Returns:
		particles (df(id, coord_x, coord_y, total_pixels, mask)): DataFrame de partículas filtradas.
	"""
	particles_out = particles[particles['total_pixels'] * (pixel_size[0] * pixel_size[1]) > 0.1]
	return particles_out
