import tifffile
import time
from evaluation import evaluation
import numpy as np
from ground_truth.generador_simulaciones_2 import generate_sequence
from test_desempeno.error_measures import error_measures, fix_particles_oustide
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Definición de parámetros sobre los que se va a probar:
poblaciones = []
mean = np.array([20.7, 9.6])
cov = np.array([[103.8, 21.6], [21.6, 14.5]])
vm = 3
poblacion = {
	'particles': 25,
	'mean': mean,
	'cov': cov,
	'mean_velocity': vm,
	'sigma_v': vm * 0.1,
	'sigma_theta': 10
}
poblaciones.append(poblacion)

mean = np.array([15, 6])
cov = np.array([[103.8, 21.6], [21.6, 14.5]])
vm = 5
poblacion = {
	'particles': 50,
	'mean': mean,
	'cov': cov,
	'mean_velocity': vm,
	'sigma_v': vm * 0.2,
	'sigma_theta': 15
}
poblaciones.append(poblacion)

print('Now generating ground truth sequence')
t0 = time.time()
ground_truth = generate_sequence(M=512, N=512, frames=100, sigma_r=4, poblaciones=poblaciones)
ground_truth_filtered = ground_truth[ground_truth.intensity > 50]
t1 = time.time()
print('Finished running df_ground_truth in:', t1 - t0, 's\n')

tif = tifffile.TiffFile('output/sal.tif')      		# se carga la salida del generador de secuencias
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
	for p in ground_truth_f_df.index:
		patch = Circle((ground_truth_f_df.at[p, 'y'], ground_truth_f_df.at[p, 'x']), radius=1, color='red')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/ground_truth'+str(nro_frame)+'.png', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots(1)
	ax.imshow(image, cmap='gray')
	for p in detected_f_df.index:
		patch = Circle((detected_f_df.at[p, 'y'],detected_f_df.at[p, 'x']), radius=1, color='blue')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/detected'+str(nro_frame)+'.png', bbox_inches='tight')
	plt.close()
