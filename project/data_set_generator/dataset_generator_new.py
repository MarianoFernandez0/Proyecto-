#!/usr/bin/python
# coding: latin-1

from skimage.filters import gaussian
from skimage.draw import ellipse
from tifffile import TiffWriter
from imageio import mimwrite as mp4_writer
from imageio import imwrite
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
import concurrent.futures


def make_sequence(sequence_parameters, all_population):
    '''
    Primary function that make the specific sequence

    Inputs:
        - sequence_parameters: list of parameters
            {
                path_data_out: string with the path where the data is going to be saved
                path_seq_out:  string with the path where the video is going to be saved
                M: Height of each frame
                N: Width of each frame
                frames: Total number of frames
                rgb: Boolean if the output is rgb or not
                std_blur: Standard deviation of blurring image
                std_noise_added: List of possible power of the added noise
                low_limit: Inferior limit of intensity
                extension: List of video extensions (tif, mp4, jpg)
                file_name: Name of the ouput file
                fps: frame rate when the ouput is mp4
            }
        - all_population: list of pupulation data
        TODO: Terminar de explicar las entradas de cada poblacion
    '''
    # Load the parameters from the list
    HOUSING_PATH_SEQ_DATA = "datasets/data_sequence"
    HOUSING_PATH_SEQ_OUT = "datasets/video_sequence"

    path_data_out, path_seq_out = sequence_parameters['path_data_out'], sequence_parameters['path_seq_out']
    M, N, duration = int(sequence_parameters['height']), int(sequence_parameters['width']), sequence_parameters[
        'duration']
    rgb = sequence_parameters['height']
    std_blur = sequence_parameters['std_blur']
    noise_type = sequence_parameters['noise_type']
    noise_params = sequence_parameters['noise_params']
    low_limit = sequence_parameters['low_limit']
    extension = sequence_parameters['file_format']
    file_name = sequence_parameters['file_name']
    frame_rates = sequence_parameters['frame_rate']
    seed = sequence_parameters['seed']
    resolution = sequence_parameters['resolution']

    M, N = int(M), int(N)
    if seed > 0: np.random.seed(seed)
    if not path_data_out:
        path_data_out = HOUSING_PATH_SEQ_DATA
    if not path_seq_out:
        path_seq_out = HOUSING_PATH_SEQ_OUT

    path_data_out = Path(path_data_out)
    path_seq_out = Path(path_seq_out)

    Path.mkdir(path_data_out, exist_ok=True, parents=True)
    Path.mkdir(path_seq_out, exist_ok=True, parents=True)



    for frame_rate in frame_rates:
        df_info = pd.DataFrame(columns=['id_particle', 'x', 'y', 'fluorescence', 'frame'])
        next_id = 0
        frames = int(np.round(duration * frame_rate, 0))
        time_step = 1 / frame_rate
        if not rgb:
            final_sequence = np.zeros((frames, M, N))
        else:
            final_sequence = np.zeros((frames, M, N, 3))

        final_sequence_segmented = np.zeros((frames, M, N))
        it = 0
        tot_it = len(all_population) * frames

        for population in all_population:
            particles = population['particles']
            color = population['color']
            mean, cov_mean = population['mean'] * resolution, population['cov_mean'] * resolution
            mean_velocity, std_velocity = population['vap'] * resolution, population['vap_deviation'] * resolution
            Tp = population['Tp']
            head_displ = population['head_displ']
            std_depth, mov_type = population['std_depth'], population['movement_type']
            ALH_mean, ALH_std = population['ALH_mean'] * resolution, population['ALH_std'] * resolution,
            BCP_mean, BCP_std = population['BCP_mean'], population['BCP_std']

            x = np.zeros([particles, frames])
            y = np.zeros([particles, frames])
            intensity = np.zeros([particles, frames])
            # Initial intensity vector for every particle
            intensity[:, 0] = np.random.uniform(150, 250, particles)
            id_particles = np.arange(next_id, next_id + particles)

            # Initial positions of the particles in a square of (3N, 3M)
            inf_x, sup_x, inf_y, sup_y = -N, 2 * N, -M, 2 * M
            x[:, 0] = np.random.uniform(inf_x, sup_x, particles)
            y[:, 0] = np.random.uniform(inf_y, sup_y, particles)
            # Size of the particles population
            dimensions = np.random.multivariate_normal(mean, cov_mean, particles)
            a = np.min(dimensions, axis=1)
            l = np.max(dimensions, axis=1)

            # Initial angle
            theta = np.random.uniform(-180, 180, particles)
            (BCP_freq, BCP_fase) = (np.random.normal(BCP_mean, BCP_std, particles),
                                    np.random.uniform(-np.pi, np.pi, particles))

            # Initial speed
            v = np.random.normal(mean_velocity, std_velocity, particles)

            head_x = head_y = head_angle = np.zeros(particles)

            # Each frame is created
            for f in range(frames):
                # Progress bar printing
                it += 1
                printProgressBar(it, tot_it)
                if f > 0:
                    x[:, f] = x[:, f - 1] + v * np.cos(np.radians(theta)) * time_step
                    y[:, f] = y[:, f - 1] + v * np.sin(np.radians(theta)) * time_step

                    # Fuera del campo de observación, sentido opuesto
                    indexes = np.logical_or(
                        np.logical_or(np.logical_or(x[:, f] < inf_x, x[:, f] > sup_x), y[:, f] < inf_y),
                        y[:, f] > sup_y)
                    theta[indexes] *= -1
                    if head_displ and mov_type != "d":
                        head_pos = np.sin(BCP_freq * 2 * np.pi * time_step * f + BCP_fase)
                        head_pos *= np.random.normal(ALH_mean, ALH_std, particles)
                        head_x = head_pos * np.cos(np.radians(theta + 90))
                        head_y = head_pos * np.sin(np.radians(theta + 90))
                        head_angle = np.pi / 4 * np.sin(BCP_freq * 2 * np.pi * time_step * f + BCP_fase)

                image_aux = final_sequence[f, :, :].copy()
                image_segmented = final_sequence_segmented[f, :, :].copy()
                # Each particle is added
                for p in range(particles):
                    rr, cc = ellipse(x[p, f] + head_x[p],
                                     y[p, f] + head_y[p], l[p], a[p], image_aux.shape,
                                     np.radians(theta[p]) + head_angle[p])
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
                    if not rgb:
                        image_aux[rr, cc] = np.where(image_aux[rr, cc] < intensity[p, f], intensity[p, f],
                                                 image_aux[rr, cc])
                    else:
                        image_aux[rr, cc] = np.where(image_aux[rr, cc] < intensity[p, f], intensity[p, f],
                                                     image_aux[rr, cc]) * color
                    # Agrego aquellas que entran en el cuadro
                    if 0 < x[p, f] < M and 0 < y[p, f] < N and intensity[p, f] > low_limit:
                        df_info = df_info.append(
                            {'id_particle': id_particles[p], 'x': y[p, f] + head_y[p], 'y': x[p, f] + head_x[p],
                             'fluorescence': intensity[p, f], 'frame': f},
                            ignore_index=True)
                    else:
                        id_particles[p] = np.max(id_particles) + 1
                # Add blur so there are no drastic changes in the border of the particles
                if not rgb:
                    image_normalized = gaussian(image_aux, std_blur, truncate=3)
                else:
                    image_normalized = gaussian(image_aux, std_blur, truncate=3, multichannel=True)
                final_sequence_segmented[f, :, :] = np.uint8(image_segmented)
                image_normalized = image_normalized.clip(0, 255)
                if not rgb:
                    final_sequence[f] = np.uint8(image_normalized)
                else:
                    final_sequence[f] = np.uint8(image_normalized)

                # Next step
                if mov_type == "d":
                    continue
                else:
                    theta += np.random.normal(0, 1, theta.shape) * np.sqrt(2 / Tp) * time_step
            next_id = np.max(id_particles)

        final_sequence *= 255 / np.max(final_sequence)

        it = 0
        tot_it = len(noise_params)
        for param in noise_params:
            it += 1
            print("Saving %d/%d of sequence with noise added..." % (it, tot_it), end="\r")
            sequence_plus_noise = add_noise(final_sequence, param)
            save_video_file(np.uint8(sequence_plus_noise), extension,
                            file_name + "_noise_added_" + str(param).replace(".", "_"), path_seq_out,
                            frame_rate)
            print() if it == tot_it else False

        final_sequence = np.uint8(final_sequence)
        print("Saving sequence without noise...")
        save_video_file(final_sequence, extension, file_name + ('(%dHz)' % frame_rate), path_seq_out,
                            frame_rate)

        print("Saving segmented sequence...")
        save_video_file(np.uint8(final_sequence_segmented), extension,
                        file_name + ('(%dHz)' % frame_rate) + "_segmented_", path_seq_out, frame_rate)

        save_data_file(df_info, str(path_data_out), file_name + ('(%dHz)' % frame_rate))

    return 0


def add_noise(sequence_in, type='white', param=15):
    sequence_out = sequence_in.copy()
    if type == 'white':
        sequence_out += np.random.normal(param, size=sequence_in.shape)
        sequence_out.clip(0, 255)
    elif type == 's&p':
        s_vs_p = 0.5
        amount = param
        # Salt
        num_salt = np.ceil(amount * sequence_out.shape * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in sequence_out.shape]
        sequence_out[coords] = 1

        # Pepper
        num_pepper = np.ceil(amount * sequence_out.shape * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in sequence_out.shape]
        sequence_out[coords] = 0
    elif type == "poisson":
        vals = len(np.unique(sequence_out))
        vals = 2 ** np.ceil(np.log2(vals))
        sequence_out = np.random.poisson(sequence_out * vals) / float(vals)
    return sequence_out


def printProgressBar(iteration, total, prefix='Making sequence', suffix='', decimals=1, length=20, fill="\u25AE",
                     printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '_' * (length - filledLength)
    print('\r%s %s %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
    return


def save_data_file(data_frame_in, path_data_out, file_name):
    """
    Save the data in csv file

    Inputs:
        - data_frame_in: DataFrame with the data
        - path_data_out: Directory where the data will be saved
    """
    data_frame_in.to_csv(path_data_out + "/" + file_name + "_data.csv", index=False)
    return


def save_video_file(sequence, extensions, file_name, path_out, fps=None):
    '''
    Save a numpy array as specific format video

    Inputs:
     - sequence: numpy array (frames, M, N)
     - extensions: list of the extensions (tiff, mp4, jpg)
     - file_name: string with the name of the file
     - path_out path to the directory where the output is going to be saved
     - fps: if the format is mp4, is necessary to specify the fps. Type float
    '''
    path_out = Path(path_out)
    for extension in extensions:
        if extension == "tif":
            path_out_tiff = Path.joinpath(path_out, "tiff_output")
            Path.mkdir(path_out_tiff, exist_ok=True)
            path_out_tiff = Path.joinpath(path_out_tiff, file_name + ".tif")
            # Guardo como tiff
            with TiffWriter(str(path_out_tiff), bigtiff=True) as tif:
                for frame in range(sequence.shape[0]):
                    tif.save((sequence[frame]))
        elif extension == "mp4":
            path_out_mp4 = Path.joinpath(path_out, "mp4_output")
            Path.mkdir(path_out_mp4, exist_ok=True)
            mp4_writer(str(path_out_mp4) + "/" + file_name + ".mp4", sequence, fps=fps)
        elif extension == "jpg":
            path_out_jpg = Path.joinpath(path_out, "jpg_output")
            Path.mkdir(path_out_jpg, exist_ok=True)
            for frame in range(sequence.shape[0]):
                imwrite(str(path_out_jpg) + file_name + "{0:04d}.jpg".format(frame), sequence[frame], format="jpg")

    return


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
        TODO: Terminar de explicar la data de las poblaciones
    '''

    config = configparser.ConfigParser()
    config.read(path)
    config.sections()

    print("Reading configuration file...")
    # Parameters
    sequence_parameters = {}
    sequence_parameters['path_data_out'] = config["Output path"]["PATH_OUTPUT_DATA_CSV"]
    sequence_parameters['path_seq_out'] = config["Output path"]["PATH_OUTPUT_DATA_SEQ"]
    (sequence_parameters['height'], sequence_parameters['width'], sequence_parameters['duration']) = \
        np.array(config["Sequence parameters"]["height_width_duration"].split(), dtype=np.float)
    sequence_parameters['rgb'] = (config["Sequence parameters"]["rgb"]).lower() == "true"
    sequence_parameters['std_blur'] = float(config["Sequence parameters"]["std_blur"])
    sequence_parameters['noise_type'] = config["Sequence parameters"]["noise_params"].split()
    sequence_parameters['noise_params'] = np.array(config["Sequence parameters"]["noise_params"].split(),
                                                   dtype=np.float)
    sequence_parameters['low_limit'] = float(config["Sequence parameters"]["low_limit"])
    sequence_parameters['file_format'] = config["Sequence parameters"]["file_format"].split()
    sequence_parameters['file_name'] = config["Sequence parameters"]["file_name"]
    sequence_parameters['frame_rate'] = np.array(config["Sequence parameters"]["frame_rate"].split(), dtype=np.float)
    sequence_parameters['seed'] = int(config["Sequence parameters"]["seed"])
    sequence_parameters['resolution'] = float(config["Sequence parameters"]["resolution"])

    # load populations
    all_population = []
    for pop in ((config.sections())[2:]):
        population = {
            'particles': int(config[pop]["tot_particles"]),
            'color': np.array(config[pop]["color"].split(), dtype=np.float),
            'mean': np.array(config[pop]["mean"].split(), dtype=np.float),
            'cov_mean': (np.array(config[pop]["cov_mean"].split(), dtype=np.float)).reshape(2, 2),
            'vap': float(config[pop]["VAP"]),
            'vap_deviation': float(config[pop]["VAP_deviation"]),
            'Tp': float(config[pop]["Tp"]),
            'head_displ': (config[pop]["head_displ"]).lower() == "true",
            'std_depth': float(config[pop]["std_depth"]),
            'movement_type': (config[pop]["movement_type"]).lower(),
            'ALH_mean': float(config[pop]["ALH_mean"]),
            'ALH_std': float(config[pop]["ALH_std"]),
            'BCP_mean': float(config[pop]["BCF_mean"]),
            'BCP_std': float(config[pop]["BCF_std"])
        }
        all_population.append(population)
    return sequence_parameters, all_population
