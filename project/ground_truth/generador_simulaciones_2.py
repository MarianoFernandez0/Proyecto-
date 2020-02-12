#!/usr/bin/python
# coding: latin-1


from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import gaussian
from skimage.draw import ellipse
from skimage.external.tifffile import TiffWriter
import numpy as np
import math

plt.rcParams['figure.figsize'] = (20.0, 15.0)
plt.rcParams['image.cmap'] = 'gray'


def generate_sequence(M = 512, N = 512, frames = 250, mean = [10, 5],
	cov = [[103, 21], [21, 14]], vm = 3, sigma_v = 3, sigma_theta = 10,
	particles = 500, sigma_r = 1, add_to_sequence = False):
	'''
	Genera una secuencia simulada con las características dadas y la guarda en la carpeta "Simulated".
	Parametros:
		M (int): Largo de las imágenes (pixeles).
		N (int): Ancho de las imágenes (pixeles).
		frames (int): Cantidad de cuadros de la secuencia.
		mean (array(2)): Media del largo y ancho de las partículas (pixeles).
		cov (array(2,2)): Matriz de covarianza del largo y ancho de las partículas.	
		vm (int): Velocidad media de las partículas (pixeles por cuadro).
		sigma_v (int): Desviación estándar del cambio en velocidad de las particulas.
		sigma_theta (int): Desviación estándar del cambio en el ángulo de las partículas.
		particles (int): Cantidad de partículas a generar.
		sigma_r (int): Desviación estándar del ruido a agregar en la imagen.
		add_to_sequence (Boolean): Si es verdadero se agrega el grupo de partículas a una secuencia existente, en lugar de generar una nueva.

	Retotorna:
		x (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
		y (array(particles,frames)): Posición en el eje x de las partículas en cada cuadro.
	'''

	I = np.zeros([M,N], dtype = "uint16")
	x = np.zeros([particles, frames])
	y = np.zeros([particles, frames])
	intensity = np.random.normal(150, 50, particles)
	final_sequence = np.zeros((M, N, frames))

	x[:, 0] = np.random.uniform(-M, 2 * M, particles)                # Posición inicial de las partículas
	y[:, 0] = np.random.uniform(-N, 2 * N, particles)    

	d = np.random.multivariate_normal(mean, cov, particles)       # Se inicializa el tamaño de las partículas
	a = d[:, 0]
	l = d[:, 1]

	theta = np.random.uniform(0, 360, particles)       # Ángulo inicial 
	v = np.random.normal(vm, 10, particles)            # Velocidad inicial

	for f in range(frames):                          # Se crean los cuadros de a uno 
	    if f > 0:
	        x[:, f] = x[:, f - 1] + v * np.cos(np.radians(theta))
	        y[:, f] = y[:, f - 1] + v * np.sin(np.radians(theta))
	    name = 'output/imagen' + str(f) + '.png'
	    if add_to_sequence:
	        Iaux = imread(name)
	    else:
	        Iaux = I.copy()

	    for p in range(particles):                  					# Se agregan las partículas a la imágen de a una
	        rr, cc = ellipse(x[p, f], y[p, f], l[p], a[p], I.shape,np.radians(theta[p]) - math.pi / 2)
	        intensity[p] = intensity[p] + np.random.normal(0,10)
	        Iaux[rr,cc] = intensity[p] 
	                
	    #Agrego blur al frame para que no sean drásticos los cambios de intesidad
	    Iblur = gaussian(Iaux, 5, mode='reflect')

	    Inoise = Iblur #+ np.random.normal(0,sigma_r,Iblur.shape) 			# Se agrega ruido a las imágenes
	    Inormalized = np.uint16(np.round(((Inoise-np.min(Inoise))/(np.max(Inoise)-np.min(Inoise))*255))) 
	    #imsave(name, np.uint8(np.round(((Inoise - np.min(Inoise)) / (np.max(Inoise) - np.min(Inoise)) * 255 ))))
	    imsave(name, Inormalized)
	    final_sequence[:, :, f] = Inormalized
	    v = np.abs(np.random.normal(v, sigma_v,particles))     #Próximo paso  
	    theta = np.random.normal(theta, sigma_theta, particles)

	    #Guardo como tiff
	    #imwrite('output/salida.tif', np.uint8(final_sequence), photometric='minisblack')
	with TiffWriter('output/salida.tif', bigtiff=True) as tif:
		for frame in range(frames):
			tif.save(final_sequence[:, :, frame], photometric='minisblack', resolution=(M,N))


	return x, y


#Hago el gradiente para que quede con un brillo parecido al de las imágenes de fmed
def make_gradient_v(width, height, h, k, a, b, theta):
    # Precalculate constants
    st, ct =  math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2

    # Generate (x,y) coordinate arrays
    y,x = np.mgrid[-k:height-k,-h:width-h]
    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)

    return np.clip(1.0 - weights, 0, 1)


def velocity(M, N, x, y):
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

def total_distance(M, N, x, y):
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

def mean_velocity(vel):
	"""
	Calcula la velocidad media de cada partícula a partir de sus velocidades instantáneas.

	Parametros:
		vel (array(particles,frames)): velocidad intantánea para cada partícula y cada cuadro de la secuencia.

	Retotorna:
		vel_m (array(particles)): Velocidad media de cada particula.

	"""
	vel_m = np.zeros(vel.shape[0])
	for p in range(vel.shape[0]):
	    count = 0
	    for f in range(vel.shape[1]):
	        if vel[p, f] != None and (math.isnan(vel[p, f]) == False):
	            vel_m[p] = vel_m[p] + vel[p, f]
	            count = count + 1
	    vel_m[p] = vel_m[p] / count
	return vel_m

mean = np.array([20.7247332, 9.61818939])
cov = np.array([[103.80124818, 21.61793687],
				 [ 21.61793687, 14.59060681]])
x, y= generate_sequence(frames = 100, sigma_r = 4, particles = 100, mean = mean, cov = cov)


#mean = np.array([10, 5])
#cov = np.array([[103.80124818, 21.61793687],
#[ 21.61793687, 14.59060681]])
#x, y= genetate_sequence(frames = 40, sigma_r = 10, particles = 100, mean = mean, cov = cov, add_to_sequence = True, vm = 7, sigma1 = 1, sigma2 = 2)

#vel = velocity(x, y, 512, 512)
#vel_m = mean_velocity(vel)
#dis = total_distance(x, y, 512, 512)
