import os
import json
import argparse
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from imageio import mimwrite
from tool.src.vis.draw_tracks import draw_tracks
from tool.src.error_measures.error_measures import track_set_error


def compute_errors(dataset_folder, dataset_path, results_file):
    performance_measures = pd.DataFrame()

    print('dataset_path: ', dataset_path)

    configs_dir = os.path.join('../../Datasets', dataset_path, 'tracking_configs')
    for config_file in tqdm(os.listdir(configs_dir)):
        if '_NN' not in config_file:
            continue
        print('-----------------------------------------------------------------------------------')
        print('config_file: ', os.path.join(configs_dir, config_file))

        # load info
        with open(os.path.join(configs_dir, config_file), 'r') as f:
            config = json.load(f)
        # read info

        frame_rate = config['fps']
        tracks_file = config['tracks_csv'].split(sep='/')
        tracks_file[5] = 'hungarian_results'
        tracks_file = os.path.join(*tracks_file)

        gt_files = os.listdir(os.path.join('../../Datasets', dataset_path, 'datasets/data_sequence'))
        gt_file = [file for file in gt_files if str(frame_rate) + 'Hz' in file][0]
        gt_file = os.path.join('../../Datasets', dataset_path, 'datasets/data_sequence', gt_file)
        # load dataframes
        tracks = pd.read_csv(os.path.join(dataset_folder, tracks_file))
        tracks = tracks[tracks['frame'] < tracks['frame'].max()]
        gt_tracks = pd.read_csv(gt_file)
        gt_tracks = gt_tracks[gt_tracks['frame'] < tracks['frame'].max()]
        gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
        # gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)

        # compute error measures
        error = track_set_error(gt_tracks, tracks, max_dist=40)
        col_name = str(frame_rate) + 'Hz'
        error = pd.DataFrame.from_dict(error, orient='index', columns=[col_name])
        error = error.transpose()
        performance_measures = pd.concat((performance_measures, error))
    os.makedirs(os.path.join(results_file.split('/')[0], results_file.split('/')[1]), exist_ok=True)
    print('results_file', results_file)
    performance_measures.to_csv(results_file)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Compute tracking performance measures')
    # parser.add_argument('--dataset_dir', default='python_plus_octave/datasets_25_08_2020/dataset_1',
    #                     help='directory of the dataset')
    # parser.add_argument('--results_file', default='error_results/dataset_performance_measures_results.csv',
    #                     help='file to save performance results')
    # parser.add_argument('--save_vid', default=True, help='Flag')
    # args = parser.parse_args()
    # compute_errors(args.dataset_dir, args.results_file, args.save_vid)
    DATASET_FOLDER = "."
    DATASET_DIRS = ["dataset_29_8_2020/dataset_1",
                    "dataset_29_8_2020/dataset_2",
                    "dataset_29_8_2020/dataset_3",
                    "dataset_29_8_2020/dataset_4",
                    "dataset_29_8_2020/dataset_5",
                    "dataset_29_8_2020/dataset_6"]

    RESULTS_FILES = ["error_results/dataset_29_8_2020_hungarian/dataset_1_performance_measures.csv",
                     "error_results/dataset_29_8_2020_hungarian/dataset_2_performance_measures.csv",
                     "error_results/dataset_29_8_2020_hungarian/dataset_3_performance_measures.csv",
                     "error_results/dataset_29_8_2020_hungarian/dataset_4_performance_measures.csv",
                     "error_results/dataset_29_8_2020_hungarian/dataset_5_performance_measures.csv",
                     "error_results/dataset_29_8_2020_hungarian/dataset_6_performance_measures.csv"]

    for i, dataset in enumerate(DATASET_DIRS):
        results = RESULTS_FILES[i]
        compute_errors(DATASET_FOLDER, dataset, results)
