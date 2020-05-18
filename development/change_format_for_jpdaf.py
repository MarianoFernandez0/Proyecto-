import os
import sys
import pathlib
import tifffile
import numpy as np
import pandas as pd
from skimage.io import imsave
from evaluation import evaluation


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
            frame = detections.loc[detections['frame'] == num_frame, ['x', 'y']]
            file.write(str(int(num_frame)))
            file.write(',')
            file.write(str(len(frame.index)))
            file.write(',')
            for index in frame.index:
                file.write(str(int(frame.loc[index, 'y'])))
                file.write(',')
                file.write(str(int(frame.loc[index, 'x'])))
                if index == frame.index[-1]:
                    file.write(',20,20')
                else:
                    file.write(',20,20,')
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


# current_path = os.getcwd()
# data_sequences_path = current_path + '/data_set_generator/datasets/data_sequence'
# data_sequences = os.listdir(data_sequences_path)
# data_sequences.sort()

# video_sequences_path = current_path + '/data_set_generator/datasets/video_sequence'
# video_sequences_all = os.listdir(video_sequences_path)
# video_sequences = [sequence for sequence in video_sequences_all if 'segmented' not in sequence]
# video_sequences.sort()

# if len(data_sequences) != len(video_sequences):
#     sys.exit('The number of data sequences does not match with the number of video sequences')

# se carga la secuencia generada artificialmente y el csv con su información
# for num_seq in [0]:#range(len(data_sequences)):
#     print('number of sequence:', num_seq)
#     print('Running detection for file:', video_sequences[num_seq])
#     ground_truth = pd.read_csv(data_sequences_path + '/' + data_sequences[num_seq])
#     tiff = tifffile.TiffFile(video_sequences_path + '/' + video_sequences[num_seq])
#     detected = evaluation(tiff)
#     folder_directory = 'sequences_for_jpdaf/' + video_sequences[num_seq] + '/'
#     save_secuence_as_jpgs_for_jpdaf(tiff, folder_directory)
#     write_csv_for_jpdaf(detected, folder_directory)
