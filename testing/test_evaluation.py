from evaluation import evaluation
from segmentation.segmentation import segmentation
from test_desempeno.error_measures import error_measures, fix_particles_oustide

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


# Test sobre el desempeño del código implementado hasta el punto de la detección, sobre el dataset simulado.


def detection_test(data_sequences, video_sequences, data_sequences_path, video_sequences_path):

	intensity_thresholds = range(40, 120, 10)
	test_results_dataset = pd.DataFrame(index=range(len(intensity_thresholds)),
										columns=['TP', 'FN', 'FP', 'JSC', 'intensity_threshold'])
	test_results_dataset['intensity_threshold'] = intensity_thresholds
	test_results_dataset[['TP', 'FN', 'FP']] = np.zeros((len(intensity_thresholds), 3))

	# se carga la secuencia generada artificialmente y el csv con su información
	for sequence in range(len(data_sequences)):
		print('Running detection tests for file:', video_sequences[sequence])
		ground_truth = pd.read_csv(data_sequences_path + '/' + data_sequences[sequence])
		tif = tifffile.TiffFile(video_sequences_path + '/' + video_sequences[sequence])

		# Las partículas del ground truth se pasan por un filtro de intensidad, y se comparan los resultados de la detección
		# con el ground truth filtrado con diferentes umbrales.
		test_results = pd.DataFrame(index=range(len(intensity_thresholds)),
									columns=['TP', 'FN', 'FP', 'JSC', 'intensity_threshold'])
		test_results['intensity_threshold'] = intensity_thresholds

		for intensity_threshold in intensity_thresholds:
			ground_truth_filtered = ground_truth[ground_truth.intensity > intensity_threshold]
			detected = evaluation(tif)
			TP, FN, FP, JSC = error_measures(ground_truth_df=ground_truth_filtered, detected_df=detected,
											 max_dist=60)

			test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'TP'] = TP
			test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'FN'] = FN
			test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'FP'] = FP
			test_results.loc[test_results['intensity_threshold'] == intensity_threshold, 'JSC'] = JSC

			test_results_dataset.loc[test_results['intensity_threshold'] == intensity_threshold, 'TP'] += TP
			test_results_dataset.loc[test_results['intensity_threshold'] == intensity_threshold, 'FN'] += FN
			test_results_dataset.loc[test_results['intensity_threshold'] == intensity_threshold, 'FP'] += FP

		test_results.to_csv('Test_results/detection_test_results_' + str(video_sequences[sequence]) + '_.csv')

		# gráficas para cada secuencia:
		plt.figure(figsize=(15, 20))
		plt.subplot(211)
		plt.title('Detection test results for sigma 0.0' + str(video_sequences[sequence]))
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

		plt.savefig('Test_results/Plots/detection_tests_results_' + str(video_sequences[sequence]) + '_.png',
					bbox_inches='tight')
		plt.close()

	# se calcula el JSC general
	test_results_dataset['JSC'] = (test_results_dataset['TP'] / (test_results_dataset['TP'] +
																test_results_dataset['FN'] +
																test_results_dataset['FP']))
	test_results_dataset.to_csv('Test_results/detection_test_results_dataset.csv')

	# gráficas generales:
	plt.figure(figsize=(15, 20))
	plt.subplot(211)
	plt.title('Detection test results for the dataset')
	plt.plot(intensity_thresholds, test_results_dataset['TP'], label='TP', c='b')
	plt.plot(intensity_thresholds, test_results_dataset['FN'], label='FN', c='r')
	plt.plot(intensity_thresholds, test_results_dataset['FP'], label='FP', c='g')
	plt.xlabel('Intensity threshold')
	plt.legend()
	plt.grid(True)

	plt.subplot(212)
	plt.plot(intensity_thresholds, test_results_dataset['JSC'])
	plt.grid(True)
	plt.xlabel('Intensity threshold')
	plt.ylabel('JSC')
	plt.savefig('Test_results/Plots/detection_tests_results dataset.png', bbox_inches='tight')
	plt.close()


def segmentation_tests(video_sequences, video_sequences_segmented, video_sequences_path):

	test_results_dataset = pd.Series(np.zeros(4), index=['TP', 'FN', 'FP', 'JSC'])

	# se cargan las secuencias generadas artificialmente
	for sequence in range(len(video_sequences)):
		print('Running segmentation tests for file:', video_sequences[sequence])
		tif_gt = tifffile.TiffFile(video_sequences_path + '/' + video_sequences[sequence])
		tif_segmented_gt = tifffile.TiffFile(video_sequences_path + '/' + video_sequences_segmented[sequence])

		sequence_gt = tif_gt.asarray()
		segmented_sequence_gt = tif_segmented_gt.asarray()

		# se calculan los resultados de para la segmentación de cada cuadro de las secuencias
		test_results = pd.DataFrame(index=[1], columns=['TP', 'FN', 'FP', 'JSC', 'frame'])
		for nro_frame in range(sequence_gt.shape[0]):
			frame_gt = sequence_gt[nro_frame, :, :]
			segmented_img = segmentation(frame_gt) / 255
			frame_segmented_gt = segmented_sequence_gt[nro_frame, :, :] / 255

			TP_img = segmented_img * frame_segmented_gt
			FN_img = frame_segmented_gt - TP_img
			FP_img = segmented_img - TP_img

			TP = np.sum(TP_img)
			FN = np.sum(FN_img)
			FP = np.sum(FP_img)
			JSC = TP / (TP + FN + FP)

			test_results = test_results.append({'TP': TP, 'FN': FN, 'FP': FP, 'JSC': JSC, 'frame': nro_frame},
												ignore_index=True)
		test_results.loc['Total'] = test_results.sum()
		test_results.loc['Total', ['JSC']] = (test_results.at['Total', 'TP'] / (test_results.at['Total', 'TP'] +
																			test_results.at['Total', 'FN'] +
																			test_results.at['Total', 'FP']))

		test_results.loc['Total', ['frame']] = np.NAN
		test_results.to_csv('Test_results/segmentation_test_results_' + str(video_sequences[sequence]) + '_.csv')

		test_results_dataset += test_results.loc['Total', ['TP', 'FN', 'FP', 'JSC']]		# resultados generales

	# se calcula el JSC general
	test_results_dataset['JSC'] = (test_results_dataset['TP'] / (test_results_dataset['TP'] +
																test_results_dataset['FN'] +
																test_results_dataset['FP']))
	test_results_dataset.to_csv('Test_results/segmentation_test_results_dataset.csv')

# ----------------------------------------------------------------------------------------------------------------------
current_path = os.getcwd()
data_sequences_path = current_path + '/data_set_generator/datasets/data_sequence'
data_sequences = os.listdir(data_sequences_path)
video_sequences_path = current_path + '/data_set_generator/datasets/video_sequence'
video_sequences_all = os.listdir(video_sequences_path)
video_sequences = [sequence for sequence in video_sequences_all if 'segmented' not in sequence]
video_sequences_segmented = [sequence for sequence in video_sequences_all if 'segmented' in sequence]

if len(data_sequences) != len(video_sequences):
	sys.exit('The number of data sequences does not match with the number of video sequences')

# se corren los tests para la detección
detection_test(data_sequences, video_sequences, data_sequences_path, video_sequences_path)

# se corren los tests para la segmentación
segmentation_tests(video_sequences, video_sequences_segmented, video_sequences_path)
