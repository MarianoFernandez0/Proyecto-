import argparse
import os
import time
import pandas as pd
import numpy as np
import json
import tifffile
from oct2py import octave
from imageio import mimwrite as mp4_writer
from imageio import mimread as mp4_reader
from src.evaluation import evaluation
from src.draw_tracks import draw_tracks

current_path = os.getcwd()
octave.addpath(current_path + '/src/SpermTrackingProject')


class TrackingParams:
    def __init__(self, params):

        self.video_file_tiff = params['Input']['tif_video_input']
        self.fps = int(params['Input']['fps'])
        self.px2um = float(params['Input']['px2um'])
        self.ROIx = int(params['Input']['ROIx'])
        self.ROIy = int(params['Input']['ROIy'])

        self.video_file_mp4 = params['Output']['input_video']
        self.csv_tracks = params['Output']['tracks_csv']
        self.detections_file = params['Output']['detections_csv']
        self.video_file_out = params['Output']['tracks_video']

        self.detection_algorithm = int(params['Algorithm params']['detection_algorithm'])
        self.reformat_detections_file = self.detection_algorithm

        self.mtt_algorithm = int(params['Algorithm params']['mtt_algorithm'])
        self.PG = float(params['Algorithm params']['PG'])
        self.PD = float(params['Algorithm params']['PD'])
        self.gv = float(params['Algorithm params']['gv'])

        self.save_movie = 0
        self.plot_results = 0
        self.snap_shot = 0
        self.plot_track_results = 0
        self.analyze_motility = 0


def tracking_urbano(params, save_vid):
    """
    Perform detection and tracking with the matlab urbano implementation.
    The parameters must be specified in the config_params file.
    """

    tiff = tifffile.TiffFile(params.video_file_tiff)
    sequence = tiff.asarray()
    mp4_writer(params.video_file_mp4, sequence, format='mp4', fps=params.fps)
    num_frames = sequence.shape[0]
    print('Total frames', num_frames)

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
    tracks['frame'] = tracks['frame']
    tracks = tracks[['id', 'x', 'y', 'fluorescence', 'frame']]
    tracks[['x', 'y']] = tracks[['x', 'y']] / params.px2um
    tracks.to_csv(params.csv_tracks, index=False)

    if save_vid:
        tracks_array = tracks.to_numpy()
        tracks_array[:, 3] = np.zeros(tracks_array.shape[0])
        tracks_array = tracks_array[tracks_array[:, 4] < tracks_array[:, 4].max()]
        sequence_tracks = draw_tracks(sequence, tracks_array)
        mp4_writer(params.video_file_out, sequence_tracks, format='mp4', fps=params.fps)

    return tracks


########################################################
#  START
########################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default=os.path.join('configs', 'config.json'),
                        type=str, help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', help='Save video with drawn tracks.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    config_params = TrackingParams(config)
    # print(config_params.__dict__)
    tracks_df = tracking_urbano(config_params, args.save_vid)
