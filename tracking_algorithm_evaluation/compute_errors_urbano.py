from error_measures import track_set_error
from python_plus_octave.functions.draw_tracks import draw_tracks
import os
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite as mp4_writer
import argparse
import json


# save_vid = True
# dataset_dir = 'python_plus_octave/datasets/dataset_1'
# performance_measures = pd.DataFrame()
#
# gt_data_dir = os.path.join(dataset_dir, 'datasets', 'data_sequence')
# video_dir = os.path.join(dataset_dir, 'datasets', 'video_sequence')
# tracking_results_dir = os.path.join(dataset_dir, 'tracking_results')
#
# gt_files = os.listdir(gt_data_dir)
# for gt_file in gt_files:
#     frame_rate = str.split(gt_file, sep='(')[1]
#     frame_rate = str.split(frame_rate, sep=')')[0]
#     print('gt data file:', gt_file)
#     print('frame rate: ', frame_rate)
#     gt_tracks = pd.read_csv(os.path.join(gt_data_dir, gt_file))
#
#     tracks = [track for track in os.listdir(tracking_results_dir) if frame_rate in track]
#     for track in tracks:
#         algorithm = str.split(track, sep='_')[-1]
#         algorithm = str.split(algorithm, sep='.')[0]
#         if frame_rate == '60Hz':  # and algorithm in ['NN']:
#             continue
#         print('track file: ', track)
#         print('algorithm: ', algorithm)
#         tracks_csv = pd.read_csv(os.path.join(tracking_results_dir, track))
#         tracks_csv = tracks_csv[tracks_csv['frame'] < tracks_csv['frame'].max()]
#         gt_csv = gt_tracks[gt_tracks['frame'] < tracks_csv['frame'].max()].copy()
#         gt_csv = gt_csv[gt_csv['frame'] > 1]
#         gt_csv.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
#
#         if save_vid:
#             video_file = [vid for vid in os.listdir(os.path.join(video_dir, 'tiff_output')) if frame_rate in vid and
#                           'segmented' not in vid and 'noise' not in vid][0]
#             tiff = tifffile.TiffFile(os.path.join(video_dir, 'tiff_output', video_file))
#             sequence = tiff.asarray()[:, :, :]
#             tracks_np = tracks_csv.to_numpy()
#             tracks_np[:, 3] = np.zeros(tracks_np.shape[0])
#             sequence_tracks = draw_tracks(sequence, tracks_np)
#             video_file_out = os.path.join(tracking_results_dir, 'videos', frame_rate + '_' + algorithm + '.mp4')
#             mp4_writer(video_file_out, sequence_tracks, format='mp4', fps=5)
#         print(type(gt_csv), type(tracks_csv))
#         error = track_set_error(gt_csv, tracks_csv, 40)
#         error = pd.DataFrame.from_dict(error, orient='index', columns=[frame_rate+'_'+algorithm])
#         error = error.transpose()
#         performance_measures = pd.concat((performance_measures, error))
#         performance_measures.to_csv('error_results/performance_measures_octave.csv')

if __name__ == '__main__':
    performance_measures = pd.DataFrame()
    algorithms = ['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF']

    parser = argparse.ArgumentParser(description='Compute tracking performance measures')
    parser.add_argument('--dataset_dir', default='python_plus_octave',
                        help='directory of the dataset')
    parser.add_argument('--results_file', default='dataset_1_performance_measures_results',
                        help='file to save performance results')
    parser.add_argument('--save_vid', default=True, help='Flag')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sub_dataset_dir = os.path.join('datasets', 'dataset_1')
    configs_dir = os.path.join(dataset_dir, sub_dataset_dir, 'tracking_configs')
    for config_file in os.listdir(configs_dir):
        # load info
        print(os.path.join(configs_dir, config_file))
        with open(os.path.join(configs_dir, config_file), 'r') as f:
            config = json.load(f)
        # read info
        gt_file = os.path.join('python_plus_octave', config['Input']['DATAFILE_PATH'])
        video_file_in = os.path.join('python_plus_octave', config['Input']['VIDEOFILE_TIFF_PATH'])
        videos_dir = os.path.join('python_plus_octave', 'tracking_results', 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        frame_rate = config['Input']['fps']
        tracks_file = config['Output']['CSV_TRACKS_PATH']
        algorithm = algorithms[int(config['Algorithm params']['mttAlgorithm']) - 1]
        # load dataframes
        tracks = pd.read_csv(os.path.join(dataset_dir, tracks_file))
        tracks = tracks[tracks['frame'] < tracks['frame'].max()]
        gt_tracks = pd.read_csv(gt_file)
        gt_tracks = gt_tracks[gt_tracks['frame'] < tracks['frame'].max()]
        gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
        gt_tracks.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
        print(gt_tracks.loc[gt_tracks['frame'] == 2, ['x', 'y']].head())
        gt_tracks.rename(columns={'x': 'y_temp'}, inplace=True)
        print(gt_tracks.loc[gt_tracks['frame'] == 2, ['y_temp', 'y']].head())
        gt_tracks.rename(columns={'y': 'x'}, inplace=True)
        print(gt_tracks.loc[gt_tracks['frame'] == 2, ['x', 'y_temp']].head())
        gt_tracks.rename(columns={'y_temp': 'y'}, inplace=True)
        print(gt_tracks.loc[gt_tracks['frame'] == 2, ['x', 'y']].head())

        # write video
        if args.save_vid:
            tif = tifffile.TiffFile(video_file_in)
            sequence = tif.asarray()[:, :, :]
            tracks_np = tracks.to_numpy()
            tracks_np[:, 3] = np.zeros(tracks_np.shape[0])
            sequence_tracks = draw_tracks(sequence, tracks_np)
            video_file_out = os.path.join(videos_dir, str(frame_rate) + 'Hz_' + algorithm + '.mp4')
            mp4_writer(video_file_out, sequence_tracks, format='mp4', fps=5)
        # compute error measures
        tracks = tracks[tracks['frame'] < 5]
        gt_tracks = gt_tracks[gt_tracks['frame'] < 5]
        error = track_set_error(gt_tracks, tracks, max_dist=40)
        col_name = str(frame_rate) + 'Hz_' + algorithm
        error = pd.DataFrame.from_dict(error, orient='index', columns=[col_name])
        error = error.transpose()
        performance_measures = pd.concat((performance_measures, error))
        performance_measures.to_csv(args.results_file)
