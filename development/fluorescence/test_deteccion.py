from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from deteccion import Particle, detect_particles, size_filter
import tifffile

#help(tifffile)
tif = tifffile.TiffFile('Images_in/11.tif')

x_res = tif.pages[0].tags['XResolution']
y_res = tif.pages[0].tags['YResolution']

pixel_size = [x_res.value[1]/x_res.value[0], y_res.value[1]/y_res.value[0]]


img = imread("Images_in/sample.jpg")[:,:,0]
seg_img = img.copy()
for m in range(img.shape[0]):
	for n in range(img.shape[1]):
		if img[m,n]<50:
			seg_img[m,n]=0
		else:
			seg_img[m,n]=1

particles = detect_particles(img, seg_img)
print('Total de partículas detectadas antes de filtrar:',len(particles))

fig, ax = plt.subplots(1)
ax.imshow(seg_img,cmap='gray')
for particle in particles:
	patch = Circle((particle.coord[1],particle.coord[0]), radius=1, color='red')
	ax.add_patch(patch)
plt.axis('off')
plt.savefig('Images_out/coords.png', bbox_inches='tight')

particles = size_filter(particles,pixel_size)

print('Total de partículas detectadas luego de filtrar:',len(particles))

fig, ax = plt.subplots(1)
ax.imshow(seg_img,cmap='gray')
for particle in particles:
	patch = Circle((particle.coord[1],particle.coord[0]), radius=1, color='red')
	ax.add_patch(patch)
plt.axis('off')
plt.savefig('Images_out/coords_filtered.png', bbox_inches='tight')

