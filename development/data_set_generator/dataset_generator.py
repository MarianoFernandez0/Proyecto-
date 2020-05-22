#!/usr/bin/python
# coding: latin-1

from skimage.filters import gaussian
from skimage.draw import ellipse
from skimage.external.tifffile import TiffWriter
import numpy as np
import pandas as pd
import configparser
import os
import math

HOUSING_PATH_SEQ_OUT = os.path.join("datasets", "video_sequence")
HOUSING_PATH_SEQ_DATA = os.path.join("datasets", "data_sequence")


def make_sequence(sequence_parameters, all_population):
    '''

    '''
    # Load the parameters from the list
    (path_data_out, path_seq_out, M, N, frames, rgb,
     std_blur, std_noise_added, low_limit, extension, file_name) = sequence_parameters
    fetch_output(housing_path_seq_data=HOUSING_PATH_SEQ_DATA, housing_path_seq_out=HOUSING_PATH_SEQ_OUT)
    df_info = pd.DataFrame(columns=['id_particle', 'x', 'y', 'frame', 'intensity'])
    next_id = 0
    # np.random.seed(seed)
    final_sequence = np.zeros((frames, M, N))
    final_sequence_segmented = np.zeros((frames, M, N))

    for population in all_population:
        particles, mean, cov_mean = population['particles'], population['mean'], population['cov_mean']
        mean_velocity, std_velocity, std_direction = population['mean_velocity'], population['std_velocity'], \
                                                     population['std_direction']
        head_displ_limits, std_depth = population['head_displ_limits'], population['std_depth']
        x = np.zeros([particles, frames])
        y = np.zeros([particles, frames])
        intensity = np.zeros([particles, frames])
        # Inicial intensity vector for every particle
        intensity[:, 0] = np.random.uniform(100, 250, particles)
        id_particles = np.arange(next_id, next_id + particles)

        # Inicial positions of the particles in a square of (3N, 3M)
        x[:, 0] = np.random.uniform(-N, 2 * N, particles)
        y[:, 0] = np.random.uniform(-M, 2 * M, particles)
        # Size of the particles population
        dimensions = np.random.multivariate_normal(mean, cov_mean, particles)
        a = np.max(dimensions, axis=1)
        l = np.min(dimensions, axis=1)

        theta = np.random.uniform(0, 360, particles)  # Initial angle
        v = np.random.normal(mean_velocity, std_velocity, particles)  # Initial speed

        # Each frame is created
        for f in range(frames):
            if f > 0:
                x[:, f] = x[:, f - 1] + v * np.cos(np.radians(theta))
                y[:, f] = y[:, f - 1] + v * np.sin(np.radians(theta))

            image_aux = final_sequence[f, :, :].copy()
            image_segmented = final_sequence_segmented[f, :, :].copy()
            # Each particle is added
            for p in range(particles):
                head_displ = np.radians(np.random.uniform(-head_displ_limits, head_displ_limits))
                rr, cc = ellipse(x[p, f], y[p, f], l[p], a[p], image_aux.shape,
                                 np.radians(theta[p]) - math.pi / 2 + head_displ)
                if f > 0:
                    random_int_add = np.random.normal(0, std_depth)
                    intensity[p, f] = intensity[p, f - 1] + random_int_add
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
            # Add blur so there are no drastic changes in the border of the particles
            image_normalized = gaussian(image_aux, std_blur, mode='reflect', preserve_range=True)
            final_sequence_segmented[f, :, :] = np.uint8(image_segmented)
            final_sequence[f, :, :] = np.uint8(image_normalized)
            # Next step
            v = np.abs(np.random.normal(v, std_velocity, particles))
            theta = np.random.normal(theta, std_direction, particles)
        next_id = np.max(id_particles)

    for std_noise in std_noise_added:
        sequence_plus_noise = final_sequence.copy()
        sequence_plus_noise += np.random.normal(0, std_noise, size=final_sequence.shape)
        save_video_file(sequence_plus_noise, extension,
                        file_name + "_noise_added_" + str(std_noise).replace(".", "_"), path_seq_out)

    save_video_file(final_sequence, extension, file_name, path_seq_out)
    save_video_file(final_sequence_segmented, extension, file_name + "_segmented_", path_seq_out)
    save_data_file(df_info, path_data_out, file_name)

    return 0


def save_video_file(sequence, extensions, file_name, path_out):
    '''
    sequence (frames, M, N)

    '''
    print("'Saving'")
    error = False
    for extension in extensions:
        if extension == "tiff":
            # Guardo como tiff
            with TiffWriter(HOUSING_PATH_SEQ_OUT + "/" + file_name + "." + extension, bigtiff=True) as tif:
                for frame in range(sequence.shape[0]):
                    tif.save((sequence[frame]))
    return error


def save_data_file(data_frame_in, path_data_out, file_name):
    error = False
    data_frame_in.to_csv(HOUSING_PATH_SEQ_DATA + "/" + file_name + "_data.csv", index=False)

    return error


def read_parameters(path='config.txt'):
    '''
    Function that when its call, read the config file in the "path" location.

    Inputs:
        - file: string with the path to the config file.

    Output:
        - sequence_parameters: List of parameters of the final sequence
            path_data_out: location where the ouputs cvs will be saved
            path_seq_out: location where the ouputs tiff files will be saved
            M: Height of each frame
            N: Width of each frame
            frames: total number of frames that correspond to the sequence
            rgb: If the output is rgb or grayscale
            std_blur: Standard deviation of the blurring made to the output so
                    the edge of the particles are smoothed out
            std_noise_added:  list of different noises power to be added
        - all_population: list of dictionaries for each population data.
    '''

    config = configparser.ConfigParser()
    config.read(path)
    config.sections()

    # Parameters
    path_data_out = config["Output path"]["PATH_OUTPUT_DATA_CSV"]
    path_seq_out = config["Output path"]["PATH_OUTPUT_DATA_SEQ"]
    M_N_frames = np.array(config["Sequence parameters"]["height_width_frames"].split(), dtype=np.int)
    rgb = config["Sequence parameters"]["rgb"]
    std_blur = float(config["Sequence parameters"]["std_blur"])
    std_noise_added = np.array(config["Sequence parameters"]["std_noise_added"].split(), dtype=np.float)
    low_limit = float(config["Sequence parameters"]["low_limit"])
    file_format = config["Sequence parameters"]["file_format"].split()
    file_name = config["Sequence parameters"]["file_name"]
    sequence_parameters = [path_data_out, path_seq_out, M_N_frames[0], M_N_frames[1],
                           M_N_frames[2], rgb, std_blur, std_noise_added, low_limit, file_format, file_name]

    # load populations
    all_population = []
    for pop in ((config.sections())[2:]):
        population = {
            'particles': int(config[pop]["tot_particles"]),
            'mean': np.array(config[pop]["mean"].split(), dtype=np.float),
            'cov_mean': (np.array(config[pop]["cov_mean"].split(), dtype=np.float)).reshape(2, 2),
            'mean_velocity': float(config[pop]["mean_velocity"]),
            'std_velocity': float(config[pop]["std_velocity"]),
            'std_direction': float(config[pop]["std_direction"]),  # previous name was std_theta
            'head_displ_limits': float(config[pop]["head_displ_limits"]),
            'std_depth': float(config[pop]["std_depth"])
        }
        all_population.append(population)
    return sequence_parameters, all_population


def fetch_output(housing_path_seq_data=HOUSING_PATH_SEQ_DATA, housing_path_seq_out=HOUSING_PATH_SEQ_OUT):
    """
    This function takes the output files paths. If exists, does nothing, if not
    creates de directory

    Inputs:
        - housing_path_seq_data : A string with the path where is going
        to be saved the output data
        - housing_path_seq_out : A string with the path where is going
        to be saved the output sequences
    """
    if not os.path.isdir(housing_path_seq_data):
        os.makedirs(housing_path_seq_data)
    if not os.path.isdir(housing_path_seq_out):
        os.makedirs(housing_path_seq_out)
    return
