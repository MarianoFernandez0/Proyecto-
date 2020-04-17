import tifffile
import time
from evaluation import evaluation
from test_desempeno.error_measures import error_measures, fix_particles_oustide
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Test sobre el desempño del código implementado hasta el punto de la detección, sobre el dataset simulado.


# se carga la secuencia generada artificialmente y el csv con su información
for sigma in range(10):
	if sigma == 0:
		tif = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_' + str(sigma) + '.tif')
		ground_truth = pd.read_csv(
			'data_set_generator/datasets/data_sequence/salida_sigma_0_' + str(sigma) + '_data.csv')
	else:
		tif = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_0' + str(sigma) + '.tif')
		ground_truth = pd.read_csv(
			'data_set_generator/datasets/data_sequence/salida_sigma_0_0' + str(sigma) + '_data.csv')

	# Las partículas del ground truth se pasan por un filtro de intensidad, y se comparan los resultados de la detección
	# con el ground truth filtrado con diferentes umbrales.
	intensity_thresholds = range(40, 120, 10)

	test_results = pd.DataFrame(index=range(len(intensity_thresholds)), columns=['TP', 'FN', 'FP', 'JSC',
																				 'intensity_threshold'])

	test_results['intensity_threshold'] = intensity_thresholds

	for intensity_threshold in intensity_thresholds:
		ground_truth_filtered = ground_truth[ground_truth.intensity > intensity_threshold]
		print('Computing tests for sigma = 0.0' + str(sigma) + ' and intensity_threshold = ' + str(intensity_threshold))
		# print('Now detecting particles in the sequence')
		# t0 = time.time()
		detected = evaluation(tif)
		# t1 = time.time()
		# print('Finished running evaluation in:', t1-t0, 's\n')

		# print('Now computing the error measures\n')
		# t0 = time.time()
		TP, FN, FP, JSC = error_measures(ground_truth_df=ground_truth_filtered, detected_df=detected,
										 max_dist=60)
		# t1 = time.time()
		# print('Finished running error_measures in:', t1-t0, 's\n')

		'''
		print('Results for intensity_threshold = ', intensity_threshold)
		print('True Positives =', TP)
		print('False Negatives = ', FN)
		print('False Positives = ', FP)
		print('JSC =', JSC)
		'''

		test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'TP'] = TP
		test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'FN'] = FN
		test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'FP'] = FP
		test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'JSC'] = JSC
	test_results.to_csv('Results/test_results_sigma_0_0'+str(sigma)+'.csv')

	plt.figure(figsize=(15, 20))
	plt.subplot(211)
	plt.title('Test results for sigma 0.0' + str(sigma))
	plt.plot(intensity_thresholds, test_results['TP'], label='TP', c='b')
	plt.plot(intensity_thresholds, test_results['FN'], label='FN', c='r')
	plt.plot(intensity_thresholds, test_results['FP'], label='FP', c='g')
	plt.yticks(range(0, 125, 5))
	plt.xlabel('Intensity threshold')
	plt.legend()
	plt.grid(True)

	plt.subplot(212)
	plt.plot(intensity_thresholds, test_results['JSC'])
	plt.grid(True)
	plt.xlabel('Intensity threshold')
	plt.ylabel('JSC')

	plt.savefig('Results/Plots/tests_results_sigma_0_0' + str(sigma) + '.png', bbox_inches='tight')
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
		patch = Circle((ground_truth_f_df.at[particle, 'y'], ground_truth_f_df.at[particle, 'x']), radius=1,
					   color='red')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/ground_truth' + str(nro_frame) + '.png', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots(1)
	ax.imshow(image, cmap='gray')
	for particle in detected_f_df.index:
		patch = Circle((detected_f_df.at[particle, 'y'], detected_f_df.at[particle, 'x']), radius=1, color='blue')
		ax.add_patch(patch)
	plt.axis('off')
	plt.savefig('Images_out/detected' + str(nro_frame) + '.png', bbox_inches='tight')
	plt.close()
'''