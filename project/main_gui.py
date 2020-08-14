import json
import os
import time

import numpy as np
import pandas as pd
import tifffile
from imageio import mimwrite as mp4_writer

from src.add_fluorescence import add_fluorescence_to_tracks
from src.draw_tracks import draw_tracks
from src.evaluation import evaluation
import PySimpleGUI as sg
from src.gui import display_gui
import sys

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
    os.environ["OCTAVE_KERNEL_JSON"] = os.path.join(application_path, 'octave_kernel/kernel.json')
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
import oct2py

os.makedirs(os.path.join(application_path, 'tmp'), exist_ok=True)
octave = oct2py.Oct2Py(temp_dir=os.path.join(application_path, 'tmp'))

octave.addpath(os.path.join(application_path, 'oct2py'))
octave.addpath(os.path.join(application_path, 'src/SpermTrackingProject'))


class TrackingParams:
    """
    Attributes:
        video_file_tiff (str): Video sequence, in .tif format.
        fps (int): Frame frequency.
        px2um (float): scale of the image.
        ROIx (int): Horizontal region of interest.
        ROIy (int): Vertical region of interest.
        video_file_mp4 (str): Input video sequence in .mp4 format.
        detections_file (str): Output csv with estimated detections.
        csv_tracks (str): Output csv with estimated tracks.
        video_file_out (str): Video with estimated tracks drawn.
        detection_algorithm (int): Detection algorithm.
                                    0 = MatLab implementation
                                    1 = Python implementation
        reformat_detections_file (int): Depends on the detection algorithm implementation.
                                            0 = MatLab implementation
                                            1 = Python implementaadd_fluorescence_to_trackstion
        mtt_algorithm (int): Multi-Target Tracking algorithm.
                                1 = NN
                                2 = GNN
                                3 = PDAF
                                4 = JPDAF
                                5 = ENN-JPDAF
                                6 = Iterated Multi-assignment
        mtt_algorithm (int): Multi-Target Tracking algorithm.
        PG (float): Prob. that a detected target falls in validation gate
        PD (float): Prob. of detection
        gv (float): Velocity Gate (um/s)

    """

    def __init__(self, params):
        self.video_file_tiff = params['tif_video_input']
        self.fps = int(params['fps'])
        self.px2um = float(params['px2um'])
        self.ROIx = int(params['ROIx'])
        self.ROIy = int(params['ROIy'])

        output_folder = params['output']
        self.video_file_mp4 = os.path.join(output_folder, 'input.mp4')
        self.detections_file = os.path.join(output_folder, 'detections.csv')
        self.csv_tracks = os.path.join(output_folder, 'tracks.csv')
        self.video_file_out = os.path.join(output_folder, 'tracks.mp4')

        self.detection_algorithm = int(params['detection_algorithm'])
        self.reformat_detections_file = self.detection_algorithm

        self.mtt_algorithm = int(params['mtt_algorithm'])
        self.PG = float(params['PG'])
        self.PD = float(params['PD'])
        self.gv = float(params['gv'])

        # opcionales del código de matlab
        self.save_movie = 0
        self.plot_results = 0
        self.snap_shot = 0
        self.plot_track_results = 0
        self.analyze_motility = 0

        congif_log = os.path.join(output_folder, 'config_log.json')
        with open(congif_log, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


def tracking_urbano(params, save_vid):
    """
    Perform detection and tracking with the matlab urbano implementation.
    The parameters must be specified in the config_params file.
    Args:
        params (TrackingParams): Tracking configuration parameters.
        save_vid (bool).
    """

    tiff = tifffile.TiffFile(params.video_file_tiff)
    sequence = tiff.asarray()
    mp4_writer(params.video_file_mp4, sequence, format='mp4', fps=params.fps)
    num_frames = sequence.shape[0]

    # Perform detection step
    print('Running detection: ')
    start = time.time()
    if params.detection_algorithm:
        # Python implementation for segmentation and detection
        detected = evaluation(tiff, params.px2um)
        detected.to_csv(params.detections_file)
    else:
        # Urbano matlab implementation for segmentation and detection
        octave.Detector(params.detections_file, params.video_file_mp4, num_frames)
        detected = pd.read_csv(params.detections_file)
    end = time.time()
    print('Time to run detection: ', end - start)

    # Perform tracking step
    print('Running tracking: ')
    start = time.time()
    octave.Tracker(params.detections_file, params.video_file_mp4, params.video_file_out, params.csv_tracks,
                   params.reformat_detections_file, num_frames, params.fps, params.px2um, params.ROIx, params.ROIy,
                   params.mtt_algorithm, params.PG, params.PD, params.gv, params.plot_results, params.save_movie,
                   params.snap_shot, params.plot_track_results, params.analyze_motility, nout=0)
    end = time.time()
    print('Time to run tracking: ', end - start)
    octave.clear_all(nout=0)

    # reformat csv_tracks file
    tracks = pd.read_csv(params.csv_tracks)
    tracks.columns = ['id', 'x', 'y', 'frame']
    tracks['fluorescence'] = np.nan
    tracks = tracks[['id', 'x', 'y', 'fluorescence', 'frame']]
    tracks[['x', 'y']] = tracks[['x', 'y']] / params.px2um

    # fluorescence
    print('Running fluorescence: ')
    start = time.time()
    tracks = add_fluorescence_to_tracks(detected, tracks)
    end = time.time()
    print('Time to run fluorescence: ', end - start)

    tracks.to_csv(params.csv_tracks, index=False)
    if save_vid:
        tracks_array = tracks.to_numpy()
        tracks_array[np.isnan(tracks_array)] = 0
        tracks_array = tracks_array[tracks_array[:, 4] < tracks_array[:, 4].max()]
        sequence_tracks = draw_tracks(sequence, tracks_array)
        mp4_writer(params.video_file_out, sequence_tracks, format='mp4', fps=params.fps)

    return tracks


########################################################
#  START
########################################################


if __name__ == '__main__':
    event, values = display_gui()
    if event not in (sg.WIN_CLOSED, 'Cancel'):
        config_params = TrackingParams(values)
        tracks_df = tracking_urbano(config_params, values['save_vid'])
