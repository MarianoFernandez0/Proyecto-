#!/usr/bin/python
# coding: latin-1

from skimage.filters import gaussian
from skimage.draw import ellipse
from skimage.external.tifffile import TiffWriter
import numpy as np
import pandas as pd
import math


###############################################################################
#			Función principal del generador de secuencias
###############################################################################
def generate_sequence(M = 512, N = 512, frames = 40, sigma_r = 1, poblaciones = []):
	'''
	Función principal del módulo que guarda en el directorio "ouput" la secuencia generada como
	un tipo de archivo tff.
	INPUT:


	OUTPUT:
		traks: df de pandas que contiene el id de particula, posición en x, posición en y
			   y la velocidad instantánea
		traks_info: df que contiene el id de la particula, la distancia total y la velocidad media.
	'''
	x, y, intensity = _make_sequence(M, N, frames, sigma_r, poblaciones)
	df = _make_coordinate_structure(x, y, intensity, M, N)

	return df

##############################################################################################
##############################################################################################


##############################################################################################
#				GENERADOR DE SECUENCIAS
##############################################################################################
def _make_sequence(M, N, frames, sigma_r, poblaciones):
	'''
	Genera una secuencia simulada con las características dadas y la guarda en la carpeta "Simulated".
	Parametros:
		M (int): Largo de las imágenes (pixeles).
		N (int): Ancho de las imágenes (pixeles).
		frames (int): Cantidad de cuadros de la secuencia.
		sigma_r (int): Desviación estándar del ruido a agregar en la imagen.
		poblaciones: lista que contiene diccionarios con la información de cada población que se desea agregar
		a la simulación.
		El diccionario tiene la siguiente estructura:
			mean (array(2)): Media del largo y ancho de las partículas (pixeles).
			cov (array(2,2)): Matriz de covarianza del largo y ancho de las partículas.	
			vm (int): Velocidad media de las partículas (pixeles por cuadro).
			sigma_v (int): Desviación estándar del cambio en velocidad de las particulas.
			sigma_theta (int): Desviación estándar del cambio en el ángulo de las partículas.
			particles (int): Cantidad de partículas a generar.
	Retotorna:
		x (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
		y (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
	'''

	image = np.zeros([M,N], dtype = "uint8")
	for poblacion in poblaciones:
		particles, mean, cov = poblacion['particles'], poblacion['mean'], poblacion['cov'] 
		vm, sigma_v, sigma_theta = poblacion['mean_velocity'], poblacion['sigma_v'], poblacion['sigma_theta']
		x = np.zeros([particles, frames])
		y = np.zeros([particles, frames])
		intensity = np.zeros([particles, frames])
		intensity[:, 0] = np.random.normal(200, 40, particles)
		final_sequence = np.zeros((M, N, frames))
		final_sequence_segmented = np.zeros((M, N, frames))

		x[:, 0] = np.random.uniform(-N, 2 * N, particles)                # Posición inicial de las partículas
		y[:, 0] = np.random.uniform(-M, 2 * M, particles)    

		d = np.random.multivariate_normal(mean, cov, particles)       # Se inicializa el tamaño de las partículas
		a = np.max(d, axis = 1)
		l = np.min(d, axis = 1)

		theta = np.random.uniform(0, 360, particles)       # Ángulo inicial 
		v = np.random.normal(vm, 10, particles)            # Velocidad inicial

		for f in range(frames):                          # Se crean los cuadros de a uno 
		    if f > 0:
		        x[:, f] = x[:, f - 1] + v * np.cos(np.radians(theta))
		        y[:, f] = y[:, f - 1] + v * np.sin(np.radians(theta))

		    image_aux = image.copy()
		    for p in range(particles):                  					# Se agregan las partículas a la imágen de a una
		        rr, cc = ellipse(x[p, f], y[p, f], l[p], a[p], image.shape, np.radians(theta[p]) - math.pi / 2)
		        intensity[p, f] = intensity[p, f] + np.random.normal(0,12)
		        if intensity[p, f] > 0 and intensity[p, f] <= 255:
		        	image_aux[rr,cc] = intensity[p, f]
		        elif intensity[p, f] < 0:
		        	image_aux[rr,cc] = 0
		        elif intensity[p, f] > 255:
		        	image_aux[rr,cc] = 255
        
		    #Agrego blur al frame para que no sean drásticos los cambios de intesidad
		    blured = gaussian(image_aux, 6, mode='reflect')
		    np.seterr(divide='ignore', invalid='ignore')
		    image_normalized = np.uint8(np.round(((blured - np.min(blured)) / (np.max(blured) - np.min(blured)) * 255))) 
		    intensity = np.uint8(np.round(((intensity - np.min(blured)) / (np.max(blured) - np.min(blured)) * 255)))
		    final_sequence_segmented[:, :, f] = np.uint8(image_aux) 
		    final_sequence[:, :, f] = np.uint8(image_normalized)
		    
		    #Próximo paso
		    v = np.abs(np.random.normal(v, sigma_v,particles))       
		    theta = np.random.normal(theta, sigma_theta, particles)
	#Guardo como tiff
	with TiffWriter('./output/salida.tif', bigtiff=True) as tif:
		for frame in range(frames):
			tif.save(final_sequence[:, :, frame], photometric='minisblack', resolution=(M,N))

	with TiffWriter('./output/salida_segmentada.tif', bigtiff=True) as tif:
		for frame in range(frames):
			tif.save(final_sequence_segmented[:, :, frame], photometric='minisblack', resolution=(M,N))

	return x, y, intensity
##############################################################################################
##############################################################################################


###############################################################################
#				Corregir tracks
###############################################################################
def _make_coordinate_structure(x, y, intensity, M, N):
	'''
	Toma como entrada una lista de coordenadas y devuelve en un DataFrame de pandas la información pasada.

	INPUT:
		x: x[particula, frames]. Matriz que contiene las coordenadas del eje horizontal.
		y: y[particula, frames]. Matriz que contiene las coordenadas del eje vertical.
	
	OUTPUT:
		DataFrame de la forma id_particula|x|y

	'''
	check_matrix = (x > 0) * (y > 0) * (x < N) * (y < M) 
	total_frames = check_matrix.shape[1]
	total_particles = check_matrix.shape[0]

	df = pd.DataFrame(columns = ['id_particle', 'x', 'y', 'frame', 'intensity'])

	id_par = 1
	seguido = False
	for p in range(total_particles):
		for f in range(total_frames - 1):
			last = total_frames - 1 == f + 1
			if (not seguido) and check_matrix[p, f]:
				id_par += 1
				df = df.append({'id_particle': id_par, 'x': x[p, f], 'y': y[p,f], 'frame': f, 'intensity': intensity[p, f]}, ignore_index = True)
				if check_matrix[p, f + 1]:
					seguido = True
			elif seguido and check_matrix[p, f]:
				df = df.append({'id_particle': id_par, 'x': x[p, f], 'y': y[p,f], 'frame': f, 'intensity': intensity[p, f]}, ignore_index = True)
				if not check_matrix[p, f + 1]:
					seguido = False
			if last and seguido:
				df = df.append({'id_particle': id_par, 'x': x[p, f + 1], 'y': y[p,f + 1], 'frame': f + 1, 'intensity': intensity[p, f + 1]}, ignore_index = True)
			elif last and check_matrix[p, f + 1]:
				id_par += 1
				df = df.append({'id_particle': id_par, 'x': x[p, f + 1], 'y': y[p,f + 1], 'frame': f + 1, 'intensity': intensity[p, f + 1]}, ignore_index = True)
		id_par += 1	
		seguido = False

	return df


###############################################################################
###############################################################################




###############################################################################
#				Velocidad
###############################################################################

def _velocity(M, N, x, y):
	"""
	Calcula la velocidad intantánea para cada partícula y cada cuadro de la secuencia.

	Parametros:
		M (int): Largo de las imágenes (pixeles).
		N (int): Ancho de las imágenes (pixeles).
	 	x (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
		y (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.

	Retotorna:
		vel (array(particles,frames)): velocidad intantánea para cada partícula y cada cuadro de la secuencia.
	"""
	vel = np.zeros(x.shape)
	for p in range(x.shape[0]):
	    for f in range(1,x.shape[1]):
	        if (x[p, f] > 0 and x[p, f] < M) and (y[p, f] > 0 and y[p, f] < N):
	            vel[p, f] = np.sqrt((x[p, f - 1] - x[p, f])**2 + (y[p, f - 1] - y[p, f])**2)
	        else: 
	            vel[p, f] = None
	return vel
###############################################################################
###############################################################################



###############################################################################
#				Distancia total
###############################################################################
def _total_distance(M, N, x, y):
	"""
	Calcula la distancia recorrida en toda la secuencia por cada particula (solo toma en cuenta cuando la partícula está en la imágen).

	Parametros:
		M (int): Largo de las imágenes (pixeles).
		N (int): Ancho de las imágenes (pixeles).
	 	x (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
		y (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.

	Retotorna:
		dis (array(particles)): Distancia recorrida en toda la secuencia por cada particula.

	"""
	dis = np.zeros(x.shape[0])
	for p in range(x.shape[0]):
	    for f in range(1, x.shape[1]):
	        if (x[p, f] > 0 and x[p, f] < M) and (y[p, f] > 0 and y[p, f] < N):
	            dis[p] = dis[p] + np.sqrt((x[p, f - 1] - x[p, f])**2 + (y[p, f - 1] - y[p, f])**2)
	return dis
###############################################################################
###############################################################################




#    Creo imagen simulada
poblaciones = []

mean = np.array([20.7247332, 9.61818939])
cov = np.array([[103.80124818, 21.61793687],
				 [ 21.61793687, 14.59060681]])
vm = 3
poblacion = {
	'particles' : 75,
	'mean' : mean,
	'cov' : cov,
	'mean_velocity' : vm,
	'sigma_v' : vm * 0.1,
	'sigma_theta' : 10
}
poblaciones.append(poblacion)

mean = np.array([15, 6])
cov = np.array([[103.80124818, 21.61793687],
[ 21.61793687, 14.59060681]])
vm = 5

poblacion = {
	'particles' : 150,
	'mean' : mean,
	'cov' : cov,
	'mean_velocity' : vm,
	'sigma_v' : vm * 0.2,
	'sigma_theta' : 15
}

poblaciones.append(poblacion)

df = generate_sequence(M = 512, N = 512, frames = 10, sigma_r = 4, poblaciones = poblaciones)

print(df)

