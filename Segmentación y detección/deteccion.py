import numpy as np
from skimage.measure import label
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

class Particle:
	count=0

	def _init_(self, id, coord, mask,total_pixels):
		self.coord = coordenates
		self.mask = mask
		Particle.count+=1
		self.id = Particle.count
		self.total_pixels = Total_of_pixels

def detect_particles(img, seg_img):
	'''
	Toma la imagen original y la segmentada, y devuelve la coordenada de las partículas

	Parametros:
		img (array(M,N)): imágen original.
		seg_img (array(M,N)): imágen segmentada.
	Returns:
		particles (list(Particle)): lista de partículas, un objeto con coordenadas x e y, y la máscara de dónde está.
	'''
	M = img.shape[0]
	N = img.shape[1]
	labeled_img, total_particles = label(seg_img,8,return_num=True)
	count = 0
	particles = []
	for m in range(M):
		for n in range(N):
			if labeled_img[m,n] == 0:
				pass
			elif labeled_img[m,n] <= count:
				particles[labeled_img[m,n]-1].coord += np.array([m,n])
				particles[labeled_img[m,n]-1].mask[m,n] == 255
				particles[labeled_img[m,n]-1].total_pixels += 1
			elif labeled_img[m,n] > count: 
				p = Particle()
				p.coord=np.array([m,n])
				p.mask=np.zeros((M,N))
				p.mask[m,n] = 255
				p.total_pixels = 1
				particles.append(p)
				count += 1

	for i in range(len(particles)):
		particles[i].coord[0] = particles[i].coord[0]/particles[i].total_pixels
		particles[i].coord[1] = particles[i].coord[1]/particles[i].total_pixels

	return particles




img = imread("sample.jpg")[:,:,0]
seg_img = img.copy()
for m in range(img.shape[0]):
	for n in range(img.shape[1]):
		if img[m,n]<50:
			seg_img[m,n]=0
		else:
			seg_img[m,n]=1

particles = detect_particles(img, seg_img)
print(len(particles))
print(particles[0].coord)

plt.imshow(seg_img,cmap='gray')
plt.plot(12,248,'r')
plt.show()

plt.imshow(particles[0].mask,cmap='gray')
plt.show()