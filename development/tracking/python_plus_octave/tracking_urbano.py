from evaluation import evaluation
import tifffile
import os
from oct2py import octave
import configparser
import pandas as pd
import numpy as np
from imageio import mimwrite as mp4_writer
from imageio import mimread as mp4_reader
import time

current_path = os.getcwd()
octave.addpath(current_path+'/SpermTrackingProject')


def tracking_urbano(config_params):
    """
    Perform detectoin and tracking with the .m urbano implementation.
    The parameters must be specified in the config_params file.
    """

    # Read params
    config_path = config_params

    config = configparser.ConfigParser()
    config.read(config_path)
    config.sections()

    data_file = config["Input"]["DATAFILE_PATH"]
    reformat_data_file = int(config["Input"]["reformat_dataFile"])

    video_file_mp4 = config["Input"]["VIDEOFILE_MP4_PATH"]
    video_file_tiff = config["Input"]["VIDEOFILE_TIFF_PATH"]
    fps = int(config["Input"]["fps"])
    px2um = float(config["Input"]["px2um"])
    ROIx = int(config["Input"]["ROIx"])
    ROIy = int(config["Input"]["ROIy"])

    csv_tracks = config["Output"]["CSV_TRACKS_PATH"]
    video_file_out = config["Output"]["VIDEOFILE_OUT_PATH"]

    detection_algorithm = int(config["Algorithm params"]["detectionAlgorithm"])
    mtt_algorithm = int(config["Algorithm params"]["mttAlgorithm"])
    PG = float(config["Algorithm params"]["PG"])
    PD = float(config["Algorithm params"]["PD"])
    gv = float(config["Algorithm params"]["gv"])

    plot_results = int(config["Do"]["plotResults"])
    save_movie = int(config["Do"]["saveMovie"])
    snap_shot = int(config["Do"]["snapShot"])
    plot_track_results = int(config["Do"]["plotTrackResults"])
    analyze_motility = int(config["Do"]["analyzeMotility"])

    # save tiff as mp4
    tiff = tifffile.TiffFile(video_file_tiff)
    sequence = tiff.asarray()
    mp4_writer(video_file_mp4, sequence, format='mp4', fps=fps)

    num_frames = sequence.shape[0]
    print('Running detection: ')
    start = time.time()
    if detection_algorithm:
        # Python implementation for segmentation and detection
        tiff = tifffile.TiffFile(video_file_tiff)
        detected = evaluation(tiff)
        detected.to_csv(data_file)
    else:
        # Urbano matlab implementation for segmentation and detection
        octave.Detector(data_file, video_file_mp4, num_frames)
    end = time.time()
    print('Time to run detection: ', end - start)
    # Perform tracking step
    start = time.time()
    octave.Tracker(data_file, video_file_mp4, video_file_out, csv_tracks, reformat_data_file, num_frames, fps, px2um,
                   ROIx, ROIy, mtt_algorithm, PG, PD, gv, plot_results, save_movie, snap_shot, plot_track_results,
                   analyze_motility, nout=0)
    end = time.time()
    print('Time to run tracking: ', end - start)
    octave.clear_all(nout=0)

    # reformat csv_tracks file
    tracks = pd.read_csv(csv_tracks)
    tracks.columns = ['id', 'x', 'y', 'frame']
    tracks['fluorescence'] = np.nan
    tracks['frame'] = tracks['frame']
    tracks = tracks[['id', 'x', 'y', 'fluorescence', 'frame']]
    tracks[['x', 'y']] = tracks[['x', 'y']]/px2um
    tracks.to_csv(csv_tracks, index=False)

    if save_movie:
        video = mp4_reader(video_file_out, memtest=False)
        mp4_writer(video_file_mp4, video, format='mp4', fps=fps//2)

########################################################
#  START
########################################################


params = 'params_real.txt'

tracking_urbano(params)
