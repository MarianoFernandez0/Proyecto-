import numpy as np
import pandas as pd
from skimage.measure import label
import matplotlib.pyplot as plt

def detect_particles(seg_img):
	'''
	Toma la imagen original y la segmentada como entrada, devuelve un dataframe con todas las partículas
	de la imagen y sus propidades.

	Parametros:
		seg_img (array(M,N)): imágen segmentada.

	Returns:
		particles (df(id, coord_x, coord_y, total_pixels, mask)): Dataframe con todas las partículas.
	'''

	M = seg_img.shape[0]
	N = seg_img.shape[1]
	labeled_img, total_particles = label(seg_img,connectivity=2,return_num=True)			#Etiqueta cada partícula con un entero diferente
	count = 0

	particles = pd.DataFrame(index = range(total_particles), columns = ['id', 'x', 'y', 'total_pixels', 'mask'])
	masks = np.zeros((M,N,total_particles))
	#Se recorren todos los pixeles de la imágen para hayar el centro geométrico de cada partícula haciendo el promedio de sus coordenadas
	#además se guardan el resto de las propiedades de las partículas
	for m in range(M):					
		for n in range(N):
			if labeled_img[m,n] == 0:
				pass
			elif pd.isna(particles.loc[labeled_img[m,n]-1,['id']]).to_numpy():
				particles.loc[labeled_img[m,n]-1,['id']] = labeled_img[m,n]
				particles.loc[labeled_img[m,n]-1,['x']]= m
				particles.loc[labeled_img[m,n]-1,['y']]= n
				masks[m,n,labeled_img[m,n]-1] = 255
				particles.loc[labeled_img[m,n]-1,['total_pixels']] = 1
			else:
				particles.loc[labeled_img[m,n]-1, ['x']] += m
				particles.loc[labeled_img[m,n]-1, ['y']] += n
				masks[m,n,labeled_img[m,n]-1] = 255
				particles.loc[labeled_img[m,n]-1,['total_pixels']] += 1

	#se divide la suma de las coordenadas sobre el total de pixeles para hayar el promedio
	print(particles.head())
	particles['x'] = particles['x']/particles['total_pixels']
	particles['y'] = particles['y']/particles['total_pixels']
	print(particles.head())
	return particles

def size_filter(particles,pixel_size):
	'''
	Toma la lista de partículas y filtra las que son menores a 10 micrometros cuadrados.

	Parámetros:
		particles (df(id, coord_x, coord_y, total_pixels, mask)): DataFrame de partículas a filtrar.
		pixel_size (list(float,float)): Las dimensiones de un pixel en micrometros.

	Retorna:
		particles (df(id, coord_x, coord_y, total_pixels, mask)): DataFrame de partículas filtradas.
	'''

	particles_out = particles[particles.loc['total_pixels']*(pixel_size[0]*pixel_size[1]) > 10]
	return particles_out

