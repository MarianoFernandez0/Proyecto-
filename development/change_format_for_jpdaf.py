import os
import sys
import pathlib
import tifffile
import numpy as np
import pandas as pd
from skimage.io import imsave
from evaluation import evaluation


def get_rectangle_detection(x, y, mask):
    """
    Devuelve el recuadro para una partícula detectada.

    Inputs:
        x (float): coordenada x del centro de la partícula.
        y (float): coordenada y del centro de la partícula.
        mask (np.array()): Máscara indicando la partícula detectada.

    Returns:
        x_new (float): coordenada x de la esquina superior izquierda de la partícula.
        y_new (float): coordenada y de la esquina superior izquierda de la partícula.
        height (float): alto del recuadro.
        width (float): ancho del recuadro.
    """
    mask_indexes = np.argwhere(mask)
    min_y = mask_indexes[:, 0].min(axis=0)
    max_y = mask_indexes[:, 0].max(axis=0)

    min_x = mask_indexes[:, 1].min(axis=0)
    max_x = mask_indexes[:, 1].max(axis=0)

    height = max_x - min_x
    width = max_y - min_y

    x_new = min_x
    y_new = min_y

    # lo que está comentado usaba el centro cálculado en a detección, pero para definir la esquina del recuadro parece
    # más adecuado lo anterior.
    # x_new = x - height/2
    # y_new = y - width/2
    # if x_new < 0:
    #     x_new = 0
    # if y_new < 0:
    #     y_new = 0

    return x_new, y_new, height, width


def write_csv_for_jpdaf(detections, csv_folder_directory):
    """
    Toma como entrada un dataframe con las detecciones y genera un .csv con el formato necesario para correr el jpdaf.
    input:
        detections: dataframe de las detecciones con las columnas "x", "y" y "frame".
        csv_folder_directory: directorio donde se guardará el .cvs
    """

    pathlib.Path(csv_folder_directory).mkdir(parents=True, exist_ok=True)
    with open(csv_folder_directory + 'detections.csv', 'w') as file:
        for num_frame in detections['frame'].unique():
            frame = detections.loc[detections['frame'] == num_frame, ['x', 'y', 'mask']]
            file.write(str(int(num_frame)))
            file.write(',')
            file.write(str(len(frame.index)))
            file.write(',')
            for index in frame.index:
                x, y, h, w = get_rectangle_detection(frame.at[index, 'x'],
                                                     frame.at[index, 'y'],
                                                     frame.at[index, 'mask'])
                file.write(str(int(y)))
                file.write(',')
                file.write(str(int(x)))
                file.write(',')
                file.write(str(h))
                file.write(',')
                file.write(str(w))
                if index == frame.index[-1]:
                    pass
                else:
                    file.write(',')
            file.write('\n')


def save_secuence_as_jpgs_for_jpdaf(tiff_sequence, sequence_folder_directory):
    """
    Toma como entrada un una secuencia en formato .tiff y la guarda en una carpeta como imágenes .jpg,
     con el formato necesario para correr el jpdaf.
    input:
        detections: dataframe de las detecciones con las columnas "x", "y" y "frame".
        sequence_folder_directory: directorio donde se guardará el .cvs
    """
    tiff_sequence = tiff_sequence.asarray()
    tiff_sequence = tiff_sequence.astype(np.uint8)
    pathlib.Path(sequence_folder_directory+'video/').mkdir(parents=True, exist_ok=True)
    for i in range(tiff_sequence.shape[0]):
        imsave(folder_directory+'video/'+str(i)+'.jpg', tiff_sequence[i, :, :])


# ----------------------------------------------------------------------------------------------------------------------
# TEST:

# current_path = os.getcwd()
# data_sequences_path = current_path + '/data_set_generator/datasets/data_sequence'
# data_sequences = os.listdir(data_sequences_path)
# data_sequences.sort()
#
# video_sequences_path = current_path + '/data_set_generator/datasets/video_sequence'
# video_sequences_all = os.listdir(video_sequences_path)
# video_sequences = [sequence for sequence in video_sequences_all if 'segmented' not in sequence]
# video_sequences.sort()
#
# if len(data_sequences) != len(video_sequences):
#     sys.exit('The number of data sequences does not match with the number of video sequences')
#
# # se carga la secuencia generada artificialmente y el csv con su información
# for num_seq in [0]:#range(len(data_sequences)):
#     print('number of sequence:', num_seq)
#     print('Running detection for file:', video_sequences[num_seq])
#     ground_truth = pd.read_csv(data_sequences_path + '/' + data_sequences[num_seq])
#     tiff = tifffile.TiffFile(video_sequences_path + '/' + video_sequences[num_seq])
#     detected = evaluation(tiff, include_mask=True)
#     folder_directory = 'sequences_for_jpdaf/' + video_sequences[num_seq] + '/'
#     save_secuence_as_jpgs_for_jpdaf(tiff, folder_directory)
#     write_csv_for_jpdaf(detected, folder_directory)

# ----------------------------------------------------------------------------------------------------------------------
