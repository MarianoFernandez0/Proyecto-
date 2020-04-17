from segmentation.segmentation import segmentation

import tifffile
import pandas as pd
import numpy as np

# se carga la secuencia generada artificialmente y su correspondiente imágen binaria

for sigma in range(10):
    if sigma == 0:
        tif_gt = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_' + str(sigma) + '.tif')
        tif_segmented_gt = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_'
                                             + str(sigma) + '_segmented.tif')
    else:
        tif_gt = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_0' + str(sigma) + '.tif')
        tif_segmented_gt = tifffile.TiffFile('data_set_generator/datasets/video_sequence/salida_sigma_0_0'
                                             + str(sigma) + '_segmented.tif')

    sequence_gt = tif_gt.asarray()
    segmented_sequence_gt = tif_segmented_gt.asarray()

    test_results = pd.DataFrame(columns=['TP', 'FN', 'FP', 'JSC', 'frame'])

    for nro_frame in range(sequence_gt.shape[0]):
        frame_gt = sequence_gt[nro_frame, :, :]
        segmented_img = segmentation(frame_gt)/255
        frame_segmented_gt = segmented_sequence_gt[nro_frame, :, :]/255

        TP_img = segmented_img*frame_segmented_gt
        FN_img = frame_segmented_gt - TP_img
        FP_img = segmented_img - TP_img

        TP = np.sum(TP_img)
        FN = np.sum(FN_img)
        FP = np.sum(FP_img)
        JSC = TP / (TP + FN + FP)

        test_results = test_results.append({'TP': TP, 'FN': FN, 'FP': FP, 'JSC': JSC, 'frame': nro_frame},
                                           ignore_index=True)

    test_results.to_csv('Results/segmentation_test_results_sigma_0_0' + str(sigma) + '.csv')