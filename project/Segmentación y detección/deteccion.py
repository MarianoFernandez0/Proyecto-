import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt

class Particle:
	#Clase que define a cada partícula detectada

	count=0

	def _init_(self, id, coord, mask,total_pixels):
		self.coord = coordenates					#Coordenadas x e y del centro geométrico de la partícula
		self.mask = mask							#Imágen con 1 dónde está la partícula y 0 en el resto de los pixeles
		Particle.count+=1		
		self.id = Particle.count					#El id es diferente para cada partícula creada
		self.total_pixels = Total_of_pixels			#Total dé pixeles que abarca la partícula

def detect_particles(img, seg_img):
	'''
	Toma la imagen original y la segmentada como entrada, devuelve una lísta con todas las partículas
	de la imagen y sus propidades.

	Parametros:
		img (array(M,N)): imágen original.
		seg_img (array(M,N)): imágen segmentada.
	Returns:
		particles (list(Particle)): lista de partículas, un objeto del tipo Particle.
	'''

	M = img.shape[0]
	N = img.shape[1]
	labeled_img, total_particles = label(seg_img,connectivity=2,return_num=True)			#Etiqueta cada partícula con un entero diferente
	count = 0
	particles = [None] * total_particles

	#Se recorren todos los pixeles de la imágen para hayar el centro geométrico de cada partícula haciendo el promedio de sus coordenadas
	#además se guardan el resto de las propiedades de las partículas
	for m in range(M):					
		for n in range(N):
			if labeled_img[m,n] == 0:
				pass
			elif particles[labeled_img[m,n]-1] != None:
				particles[labeled_img[m,n]-1].coord += np.array([m,n])
				particles[labeled_img[m,n]-1].mask[m,n] = 255
				particles[labeled_img[m,n]-1].total_pixels += 1
			else: 
				p = Particle()
				p.coord=np.array([m,n])
				p.mask=np.zeros((M,N))
				p.mask[m,n] = 255
				p.total_pixels = 1
				particles[labeled_img[m,n]-1] = p
				count += 1

	for i in range(len(particles)):		#se divide la suma de las coordenadas sobre el total de pixeles para hayar el promedio
		particles[i].coord[0] = particles[i].coord[0]/particles[i].total_pixels
		particles[i].coord[1] = particles[i].coord[1]/particles[i].total_pixels

	return particles

def size_filter(particles,k):
	'''
	Toma la lista de partículas y filtra las que se alejan "k" desviaciones estándar de la media.

	Parámetros:
		particles (list(Particle)): Lista de objetos de tipo Particle a filtrar.
		k (int/float): Las partículas que su tamaño se aleje k desviaciones estándar de la media son filtradas.

	Retorna:
		particles (list(Particle)): Lista de objetos de tipo Particle filtrada.
	'''

	N = len(particles)
	sizes = np.zeros(N)

	for n in range(N):
		sizes[n] = particles[n].total_pixels

	mean = np.mean(sizes)
	std = np.std(sizes)

	particles_out = []

	for n in range(N):
#		if (particles[n].total_pixels < mean + k*std and
#				particles[n].total_pixels > mean - k*std):
		if (particles[n].total_pixels > mean - k*std):
			particles_out.append(particles[n])


	#Test:
	fig, ax = plt.subplots(1)
	plt.hist(sizes, N//1)
	plt.axvline(x=mean, label='mean'.format(mean), c='r')

	for i in range(1,4):
		plt.axvline(x=mean+i*std, label='mean +'.format(k), c='g')
		plt.axvline(x=mean-i*std, label='mean -'.format(k), c='g')
	plt.savefig('Images_out/hist_sizes.png', bbox_inches='tight')


	return particles_out

