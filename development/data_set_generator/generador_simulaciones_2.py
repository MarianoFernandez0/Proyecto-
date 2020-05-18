#!/usr/bin/python
# coding: latin-1

from skimage.filters import gaussian
from skimage.draw import ellipse
from tifffile import TiffWriter
import numpy as np
import pandas as pd
import os
import math

###############################################################################
#			Establezco carpetas dc x<zsdfewq 21 zswq    ��e salida
###############################################################################
HOUSING_PATH_SEQ_OUT = os.path.join("datasets", "video_sequence")
HOUSING_PATH_SEQ_DATA = os.path.join("datasets", "data_sequence")


def fetch_output(housing_path_seq_data=HOUSING_PATH_SEQ_DATA, housing_path_seq_out=HOUSING_PATH_SEQ_OUT):
    if not os.path.isdir(housing_path_seq_data):
        os.makedirs(housing_path_seq_data)
    if not os.path.isdir(housing_path_seq_out):
        os.makedirs(housing_path_seq_out)
    return


###############################################################################
###############################################################################


###############################################################################
# Funcion principal del generador de secuencias
###############################################################################
def generate_sequence(M=512, N=512, frames=40, sigma_r=1, poblaciones=[], output_file_name="salida", seed=0):
    '''
    Funcion principal del modulo que guarda en el directorio "ouput" la secuencia generada como
    un tipo de archivo tff.
    INPUT:


    OUTPUT:
        traks: df de pandas que contiene el id de particula, posicion en x, posicion en y
               y la velocidad instantánea
        traks_info: df que contiene el id de la particula, la distancia total y la velocidad media.
    '''
    df = _make_sequence(M, N, frames, sigma_r, poblaciones, output_file_name, seed)

    df.to_csv(HOUSING_PATH_SEQ_DATA + "/" + output_file_name + "_data.csv", index=False)
    return


##############################################################################################
##############################################################################################


##############################################################################################
#				GENERADOR DE SECUENCIAS
##############################################################################################
def _make_sequence(M, N, frames, sigma_r, poblaciones, output_file_name, seed):
    '''
    Genera una secuencia simulada con las características dadas y la guarda en la carpeta "Simulated".
    Parametros:
        M (int): Largo de las imágenes (pixeles).
        N (int): Ancho de las imágenes (pixeles).
        frames (int): Cantidad de cuadros de la secuencia.
        sigma_r (int): Desviacion estándar del ruido a agregar en la imagen.
        poblaciones: lista que contiene diccionarios con la informacion de cada poblacion que se desea agregar
        a la simulacion.
        El diccionario tiene la siguiente estructura:
            mean (array(2)): Media del largo y ancho de las particulas (pixeles).
            cov (array(2,2)): Matriz de covarianza del largo y ancho de las particulas.
            vm (int): Velocidad media de las particulas (pixeles por cuadro).
            sigma_v (int): Desviacion estándar del cambio en velocidad de las particulas.
            sigma_theta (int): Desviacion estándar del cambio en el ángulo de las particulas.
            particles (int): Cantidad de particulas a generar.
    Retotorna:
        x (array(particles,frames)): Posicion en el eje x de las particulas en cada cuadro.
        y (array(particles,frames)): Posicion en el eje x de las particulas en cada cuadro.
    '''
    tot_particles = sum([pob['particles'] for pob in poblaciones])
    df_info = pd.DataFrame(columns=['id_particle', 'x', 'y', 'frame', 'intensity'])
    next_id = 0
    image_aux = np.zeros([M, N], dtype="uint8")
    image_segmented = np.zeros([M, N], dtype="uint8")
    low_limit = 0
    np.random.seed(seed)
    final_sequence = np.zeros((M, N, frames))
    final_sequence_segmented = np.zeros((M, N, frames))

    for poblacion in poblaciones:
        particles, mean, cov = poblacion['particles'], poblacion['mean'], poblacion['cov']
        vm, sigma_v, sigma_theta = poblacion['mean_velocity'], poblacion['sigma_v'], poblacion['sigma_theta']
        x = np.zeros([particles, frames])
        y = np.zeros([particles, frames])
        intensity = np.zeros([particles, frames])
        intensity[:, 0] = np.random.normal(180, 30, particles)
        id_particles = np.arange(next_id, next_id + particles)
        x[:, 0] = np.random.uniform(-2 * N, 4 * N, particles)  # Posicion inicial de las partículas
        y[:, 0] = np.random.uniform(-2 * M, 4 * M, particles)

        d = np.random.multivariate_normal(mean, cov, particles)  # Se inicializa el tamaño de las partículas
        a = np.max(d, axis=1)
        l = np.min(d, axis=1)

        theta = np.random.uniform(0, 360, particles)  # Angulo inicial
        v = np.random.normal(vm, 10, particles)  # Velocidad inicial

        for f in range(frames):  # Se crean los cuadros de a uno
            if f > 0:
                x[:, f] = x[:, f - 1] + v * np.cos(np.radians(theta))
                y[:, f] = y[:, f - 1] + v * np.sin(np.radians(theta))

            image_aux = final_sequence[:, :, f].copy()
            image_segmented = final_sequence_segmented[:, :, f].copy()
            for p in range(particles):  # Se agregan las partículas a la imágen de a una
                rr, cc = ellipse(x[p, f], y[p, f], l[p], a[p], image_aux.shape, np.radians(theta[p]) - math.pi / 2)
                if f > 0:
                    intensity[p, f] = np.uint8(intensity[p, f - 1] + np.random.normal(0, 2))
                if low_limit < intensity[p, f] <= 255:
                    image_segmented[rr, cc] = 255
                if intensity[p, f] <= low_limit:
                    image_aux[rr, cc] = 0
                    intensity[p, f] = 0
                if intensity[p, f] > 255:
                    intensity[p, f] = 255
                image_aux[rr, cc] = np.where(image_aux[rr, cc] < intensity[p, f], intensity[p, f], image_aux[rr, cc])

                # Agrego aquellas que entran en el cuadro
                if 0 < x[p, f] < M and 0 < y[p, f] < N and intensity[p, f] > low_limit:
                    df_info = df_info.append(
                        {'id_particle': id_particles[p], 'x': x[p, f], 'y': y[p, f], 'frame': f,
                         'intensity': intensity[p, f]},
                        ignore_index=True)
                else:
                    id_particles[p] = np.max(id_particles) + 1

            # Agrego blur al frame para que no sean drásticos los cambios de intesidad
            blured = gaussian(image_aux, 6, mode='reflect') + + np.random.normal(0, sigma_r, size=image_aux.shape)
            image_normalized = np.uint8(
                np.round(((blured - np.min(blured)) / (np.max(blured) - np.min(blured)) * 255)))
            intensity[:, f] = np.uint8(np.round(((intensity[:, f] - np.min(intensity[:, f])) / (
                    np.max(intensity[:, f]) - np.min(intensity[:, f])) * 255)))
            final_sequence_segmented[:, :, f] = np.uint8(image_segmented)
            final_sequence[:, :, f] = np.uint8(image_normalized)
            # Proximo paso
            v = np.abs(np.random.normal(v, sigma_v, particles))
            theta = np.random.normal(theta, sigma_theta, particles)

        next_id = np.max(id_particles)
    # Guardo como tiff
    with TiffWriter(HOUSING_PATH_SEQ_OUT + "/" + output_file_name + '.tif', bigtiff=True) as tif:
        for frame in range(frames):
            tif.save(final_sequence[:, :, frame], photometric='minisblack', resolution=(M, N))

    with TiffWriter(HOUSING_PATH_SEQ_OUT + "/" + output_file_name + '_segmented.tif', bigtiff=True) as tif:
        for frame in range(frames):
            tif.save(final_sequence_segmented[:, :, frame], photometric='minisblack', resolution=(M, N))

    return df_info


##############################################################################################
##############################################################################################


###############################################################################
#				Print progress
###############################################################################
# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='>', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    return


###############################################################################
###############################################################################

###############################################################################
#				Datos de poblacion
###############################################################################
populations = []

mean = np.array([18, 7])
cov = np.array([[4, 0],
                [0, 4]])
vm = 3
population = {
    'particles': 75,
    'mean': mean,
    'cov': cov,
    'mean_velocity': vm,
    'sigma_v': vm * 0.1,
    'sigma_theta': 10
}
populations.append(population)

vm = 15

population = {
    'particles': 50,
    'mean': mean,
    'cov': cov,
    'mean_velocity': vm,
    'sigma_v': vm * 0.15,
    'sigma_theta': 15
}

populations.append(population)

vm = 30
population = {
    'particles': 30,
    'mean': mean,
    'cov': cov,
    'mean_velocity': vm,
    'sigma_v': vm * 0.1,
    'sigma_theta': 30
}

populations.append(population)

###############################################################################
###############################################################################
seq_out = input("Set the output sequence path. For default directory insert '-'. Default = " + HOUSING_PATH_SEQ_OUT)
seq_data = input("Set the output data path. For default directory insert '-'. Default = " + HOUSING_PATH_SEQ_DATA)
M = int(input("Height: "))
N = int(input("Width: "))
frames = int(input("frames: "))

if not seq_out == "-":
    HOUSING_PATH_SEQ_OUT = seq_out
if not seq_data == "-":
    HOUSING_PATH_SEQ_DATA = seq_data
fetch_output(HOUSING_PATH_SEQ_OUT, HOUSING_PATH_SEQ_DATA)

sigmas_r = np.arange(0, 0.1, 0.01)
total_it = sigmas_r.shape[0]
it = 0
for sigma_r in sigmas_r:
    generate_sequence(M, N, frames, sigma_r, poblaciones=populations,
                      output_file_name="salida" + "_sigma_" + str(sigma_r).replace(".", "_"), seed=2)
    printProgressBar(it, total_it - 1, prefix='Progress:', suffix='Complete', length=50)
    it += 1
