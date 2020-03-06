import tifffile
from segmentation import segmentation
from skimage.io import imsave
import numpy as np
from skimage.external.tifffile import TiffWriter


# Leer simulacion
tif = tifffile.TiffFile('images/sal.tif')
input_sequence = tif.asarray()

# Crear stack de imagenes segmentadas
segmentation_stack = np.ones((input_sequence.shape[0],input_sequence.shape[1],input_sequence.shape[2]))
image=0

for image in range(input_sequence.shape[0]):
	segmentation_stack[image,:,:] = segmentation(input_sequence[image,:,:])

# Guardar segmentacion de imagenes de entrada 
with TiffWriter('images/segmentation_stack.tif', bigtiff=True) as tif:
		for frame in range(input_sequence.shape[0]):
			tif.save(segmentation_stack[frame,:, :], photometric='minisblack', resolution=(input_sequence.shape[1],input_sequence.shape[2]))

# Leer ground truth
tif = tifffile.TiffFile('images/salida_segmentada.tif')
ground_truth = tif.asarray()

#Comparar
matching_table = np.zeros((input_sequence.shape[0],input_sequence.shape[1],input_sequence.shape[2]))
image = 0

for image in range(input_sequence.shape[0]):
	for i in range(input_sequence.shape[1]):
		for j in range(input_sequence.shape[2]):

			if (segmentation_stack[image,i,j] == ground_truth[image,i,j]) and segmentation_stack[image,i,j]==0:
				matching_table[image,i,j] = 0
			elif (segmentation_stack[image,i,j] == ground_truth[image,i,j]) and segmentation_stack[image,i,j]==255:
				matching_table[image,i,j] = 255
			elif segmentation_stack[image,i,j]==0 and ground_truth[image,i,j]==255:
				matching_table[image,i,j]=200
			elif segmentation_stack[image,i,j]==255 and ground_truth[image,i,j]==0:
				matching_table[image,i,j]=20

			
with TiffWriter('images/matching_table.tif', bigtiff=True) as tif:
		for frame in range(input_sequence.shape[0]):
			tif.save(matching_table[frame,:, :], photometric='minisblack', resolution=(input_sequence.shape[1],input_sequence.shape[2]))

