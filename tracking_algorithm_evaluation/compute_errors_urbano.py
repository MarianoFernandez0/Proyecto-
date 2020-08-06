from error_measures import track_set_error
from python_plus_octave.functions.draw_tracks import draw_tracks
import os
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite as mp4_writer
import argparse
import json
from tqdm import tqdm


if __name__ == '__main__':
    performance_measures = pd.DataFrame()
    algorithms = ['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF']

    parser = argparse.ArgumentParser(description='Compute tracking performance measures')
    parser.add_argument('--dataset_dir', default='python_plus_octave/datasets/dataset_1',
                        help='directory of the dataset')
    parser.add_argument('--results_file', default='dataset_1_performance_measures_results',
                        help='file to save performance results')
    parser.add_argument('--save_vid', default=True, help='Flag')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.split('/')[0]
    sub_dataset_dir = os.path.join(args.dataset_dir.split('/')[1], args.dataset_dir.split('/')[2])
    configs_dir = os.path.join(dataset_dir, sub_dataset_dir, 'tracking_configs')
    for config_file in tqdm(os.listdir(configs_dir)):
        # load info
        with open(os.path.join(configs_dir, config_file), 'r') as f:
            config = json.load(f)
        # read info
        gt_file = os.path.join(dataset_dir, config['Input']['DATAFILE_PATH'])
        video_file_in = os.path.join(dataset_dir, config['Input']['VIDEOFILE_TIFF_PATH'])
        videos_dir = os.path.join(dataset_dir, sub_dataset_dir, 'tracking_results', 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        frame_rate = config['Input']['fps']
        tracks_file = config['Output']['CSV_TRACKS_PATH']
        algorithm = algorithms[int(config['Algorithm params']['mttAlgorithm']) - 1]
        # print("frame_rate: ", frame_rate)
        # print("algorithm", algorithm)
        # print("Config file: ", config_file)
        # print(os.path.join(configs_dir, config_file))
        # print("gt_file: ", gt_file)
        # print("video_file_in: ", video_file_in)
        # print("videos_dir: ", videos_dir)
        # print("tracks_file: ", tracks_file)

        # load dataframes
        tracks = pd.read_csv(os.path.join(dataset_dir, tracks_file))
        tracks = tracks[tracks['frame'] < tracks['frame'].max()]
        gt_tracks = pd.read_csv(gt_file)
        gt_tracks = gt_tracks[gt_tracks['frame'] < tracks['frame'].max()]
        gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
        gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)
        # gt_tracks.rename(columns={'x': 'y_temp'}, inplace=True)
        # gt_tracks.rename(columns={'y': 'x'}, inplace=True)
        # gt_tracks.rename(columns={'y_temp': 'y'}, inplace=True)

        # write video
        if args.save_vid:
            tif = tifffile.TiffFile(video_file_in)
            sequence = tif.asarray()[:, :, :]
            tracks_np = tracks.to_numpy()
            tracks_np[:, 3] = np.zeros(tracks_np.shape[0])
            sequence_tracks = draw_tracks(sequence, tracks_np)
            os.makedirs(videos_dir, exist_ok=True)
            video_file_out = os.path.join(videos_dir, str(frame_rate) + 'Hz_' + algorithm + '.mp4')
            print("video_file_out ", video_file_out)
            mp4_writer(video_file_out, sequence_tracks, format='mp4', fps=5)

        # compute error measures

        # tracks = tracks[tracks['frame'] < 5]
        # gt_tracks = gt_tracks[gt_tracks['frame'] < 5]
        error = track_set_error(gt_tracks, tracks, max_dist=40)
        col_name = str(frame_rate) + 'Hz_' + algorithm
        error = pd.DataFrame.from_dict(error, orient='index', columns=[col_name])
        error = error.transpose()
        performance_measures = pd.concat((performance_measures, error))
    os.makedirs(os.path.join(args.results_file.split('/')[0], args.results_file.split('/')[1]), exist_ok=True)
    performance_measures.to_csv(args.results_file)
