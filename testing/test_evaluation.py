import tifffile
import time
from evaluation import evaluation
from test_desempeno.error_measures import error_measures, fix_particles_oustide
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

tif = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0.tif')      		# se carga la salida del generador de secuencias
ground_truth = pd.read_csv('data_set_generator/datasets/data_sequence/salida_sigma_0_data.csv')
print(ground_truth.head())
ground_truth_filtered = ground_truth[ground_truth.intensity > 50]

print('Now detecting particles in the sequence')
t0 = time.time()
detected = evaluation(tif)
t1 = time.time()
print('Finished running evaluation in:', t1-t0, 's\n')

print('Now computing the error measures')
t0 = time.time()
TP, FN, FP, JSC = error_measures(ground_truth_df=ground_truth_filtered, detected_df=detected, M=512, N=512, max_dist=60)
t1 = time.time()
print('Finished running error_measures in:', t1-t0, 's\n')

print('True Positives =', TP)
print('False Negatives = ', FN)
print('False Positives = ', FP)
print('JSC =', JSC)
print('Particles in ground_truth:', ground_truth_filtered.shape[0])
print('Particles in detected:', detected.shape[0])

sequence = tif.asarray()
for nro_frame in range(sequence.shape[0]):
	image = sequence[nro_frame, :, :]
	ground_truth_f_df = ground_truth_filtered[ground_truth_filtered.frame.astype(int) == nro_frame]
	detected_f_df = detected[detected.frame.astype(int) == nro_frame]

	fig, ax = plt.subplots(1)
	ax.imshow(image, cmap='gray')
	for particle in ground_truth_f_df.index:
		patch = Circle((ground_truth_f_df.at[particle, 'y'], ground_truth_f_df.at[particle, 'x']), radius=1, color='red')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/ground_truth'+str(nro_frame)+'.png', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots(1)
	ax.imshow(image, cmap='gray')
	for particle in detected_f_df.index:
		patch = Circle((detected_f_df.at[particle, 'y'], detected_f_df.at[particle, 'x']), radius=1, color='blue')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/detected'+str(nro_frame)+'.png', bbox_inches='tight')
	plt.close()
