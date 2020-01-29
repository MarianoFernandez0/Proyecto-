from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from deteccion import Particle, detect_particles, size_filter
import cv2

img = imread("Images_in/sample.jpg")[:,:,0]
seg_img = img.copy()
for m in range(img.shape[0]):
	for n in range(img.shape[1]):
		if img[m,n]<50:
			seg_img[m,n]=0
		else:
			seg_img[m,n]=1


cv2.imwrite('out.png',seg_img) 
#plt.savefig('seg_img.png')

particles = detect_particles(img, seg_img)
mascara=particles[0].mask

cv2.imwrite('mascara.png',mascara)