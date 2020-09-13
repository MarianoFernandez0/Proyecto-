import cv2
import numpy as np
from skimage.measure import label
import pandas as pd


def gray_to_binary(img_in):
    greyscale = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)  # Convertir a grises
    # Aplicar LoG
    gaussian_blur = cv2.GaussianBlur(greyscale, (3, 3), 1.6)
    laplacian = cv2.Laplacian(gaussian_blur, cv2.CV_64F)

    positive_laplacian = np.abs(np.min(laplacian)) + laplacian  # Pasar a valores positivos
    img_inv_uint8 = np.array(positive_laplacian, dtype=np.uint8)  # Pasar a uint8
    _, otsu = cv2.threshold(img_inv_uint8, 0, np.max(img_inv_uint8),
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # aplicar otsu
    img_out = np.where(otsu == np.max(img_inv_uint8), 0, 1)  # binarizar img
    return img_out


def binary_detection(img):
    labeled_img, total_particles = label(img, connectivity=2, return_num=True)  # etiquetar particulas
    particles = pd.DataFrame(index=range(total_particles), columns=['id', 'x', 'y', 'total_pixels'])  # crear dataframe

    for p in range(1, total_particles + 1):  # rellenar dataframe
        particles.loc[p - 1, ['id']] = p
        coords_p = np.argwhere(labeled_img == p)
        particles.loc[p - 1, ['x']] = np.mean(coords_p[:, 0])
        particles.loc[p - 1, ['y']] = np.mean(coords_p[:, 1])
        particles.loc[p - 1, ['total_pixels']] = coords_p.shape[0]
    return particles


def gray_evaluation(video_in):
    df_out = pd.DataFrame(columns=['x', 'y', 'frame', 'total_pixels'])  # crear dataframe de salida
    for nro_frame in range(0, len(video_in)):
        img_in = video_in[nro_frame]  # extraer imagen del video
        binary = gray_to_binary(img_in)  # convertir la imagen a binaria
        df_particles = binary_detection(binary)  # obtener dataframe con la ubicacion de cada particula

        for index, row in df_particles.iterrows():
            df_out = df_out.append({'x': row['x'], 'y': row['y'], 'frame': nro_frame,
                                   'total_pixels': row['total_pixels']}, ignore_index=True) # rellenar dataframe
    return df_out



if __name__ == '__main__':

    import imageio
    import os

    video = imageio.mimread('1472 semen-00.avi')  # importar video

    #name_folder = 'img_binarias'
    #os.makedirs(name_folder, exist_ok=True)

    df_video = gray_evaluation(video)
    df_video.to_csv('df_video.csv', index=False, header=True)
