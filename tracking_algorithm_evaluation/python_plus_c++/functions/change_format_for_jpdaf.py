import pathlib
import numpy as np
from skimage.io import imsave


def _get_rectangle_detection(x, y, mask):
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

    x_new = min_y
    y_new = min_x

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
                x, y, h, w = _get_rectangle_detection(frame.at[index, 'x'],
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
    pathlib.Path(sequence_folder_directory + 'video/').mkdir(parents=True, exist_ok=True)
    print(sequence_folder_directory + 'video/')
    for i in range(tiff_sequence.shape[0]):
        imsave(sequence_folder_directory + 'video/{0:03d}.jpg'.format(i), tiff_sequence[i, :, :])