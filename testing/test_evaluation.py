import tifffile
import time
from evaluation import evaluation
from test_desempeno.error_measures import error_measures, fix_particles_oustide
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

# se carga la secuencia generada artificialmente y el csv con su información
tif = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_0.tif')
ground_truth = pd.read_csv('data_set_generator/datasets/data_sequence/salida_sigma_0_0_data.csv')

# Las partículas del ground truth se pasan por un filtro de intensidad, y se comparan los resultados de la detección
# con el ground truth filtrado con diferentes umbrales.
intensity_thresholds = range(40, 120, 10)

TPs = []
FNs = []
FPs = []
JSCs = []

for intensity_threshold in intensity_thresholds:
	ground_truth_filtered = ground_truth[ground_truth.intensity > intensity_threshold]

	print('Now detecting particles in the sequence')
	# t0 = time.time()
	detected = evaluation(tif)
	# t1 = time.time()
	# print('Finished running evaluation in:', t1-t0, 's\n')

	print('Now computing the error measures\n')
	# t0 = time.time()
	TP, FN, FP, JSC = error_measures(ground_truth_df=ground_truth_filtered, detected_df=detected, M=512, N=512, max_dist=60)
	# t1 = time.time()
	# print('Finished running error_measures in:', t1-t0, 's\n')

	print('Results for intensity_threshold = ', intensity_threshold)
	print('True Positives =', TP)
	print('False Negatives = ', FN)
	print('False Positives = ', FP)
	print('JSC =', JSC)

	TPs.append(TP)
	FNs.append(FN)
	FPs.append(FP)
	JSCs.append(JSC)

plt.figure(figsize=(15, 20))
plt.subplot(211)
plt.plot(intensity_thresholds, TPs, label='TP', c='b')
plt.plot(intensity_thresholds, FNs, label='FN', c='r')
plt.plot(intensity_thresholds, FPs, label='FP', c='g')
plt.yticks(range(0, 125, 5))
plt.xlabel('Intensity threshold')
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(intensity_thresholds, JSCs)
plt.grid(True)
plt.xlabel('Intensity threshold')
plt.ylabel('JSC')
plt.title('Test results')
plt.savefig('Plots/tests_results.png', bbox_inches='tight')
plt.close()

'''
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
'''