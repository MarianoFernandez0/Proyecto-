from tool.src.error_measures.error_measures import track_set_error
from tool.src.vis.draw_tracks import draw_tracks
import os
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite
import argparse
import json
from tqdm import tqdm


def compute_errors(dataset_dir, results_file, save_vid=True):
    performance_measures = pd.DataFrame()
    algorithms = ['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF']

    print('dataset_dir: ', dataset_dir)
    sub_dataset_dir = os.path.join(dataset_dir.split('/')[1], dataset_dir.split('/')[2])
    dataset_dir = dataset_dir.split('/')[0]
    configs_dir = os.path.join(dataset_dir, sub_dataset_dir, 'tracking_configs')
    for config_file in tqdm(os.listdir(configs_dir)):
        print('-----------------------------------------------------------------------------------')
        print('config_file: ', os.path.join(configs_dir, config_file))

        # load info
        with open(os.path.join(configs_dir, config_file), 'r') as f:
            config = json.load(f)
        # read info

        video_file_in = os.path.join(dataset_dir, config['Input']['tif_video_input'])

        videos_dir = os.path.join(dataset_dir, sub_dataset_dir, 'tracking_results', 'videos')

        os.makedirs(videos_dir, exist_ok=True)
        frame_rate = config['Input']['fps']
        tracks_file = config['Output']['tracks_csv']
        algorithm = algorithms[int(config['Algorithm params']['mtt_algorithm']) - 1]

        gt_files = os.listdir(os.path.join(dataset_dir, sub_dataset_dir, 'datasets/data_sequence'))
        gt_file = [file for file in gt_files if str(frame_rate) + 'Hz' in file][0]
        gt_file = os.path.join(dataset_dir, sub_dataset_dir, 'datasets/data_sequence', gt_file)

        # load dataframes
        tracks = pd.read_csv(os.path.join(dataset_dir, tracks_file))
        tracks = tracks[tracks['frame'] < tracks['frame'].max()]
        gt_tracks = pd.read_csv(gt_file)
        gt_tracks = gt_tracks[gt_tracks['frame'] < tracks['frame'].max()]
        gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
        gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)

        # write video
        if save_vid:
            tif = tifffile.TiffFile(video_file_in)
            sequence = tif.asarray()[:, :, :]
            tracks_np = tracks.to_numpy()
            tracks_np = np.where(np.isnan(tracks_np), 0, tracks_np)

            sequence_tracks = draw_tracks(sequence, tracks_np)
            os.makedirs(videos_dir, exist_ok=True)
            video_file_out = os.path.join(videos_dir, str(frame_rate) + 'Hz_' + algorithm + '.mp4')
            mimwrite(video_file_out, sequence_tracks, format='mp4', fps=5)
            print("\n video_file_out ", video_file_out)

        # compute error measures

        error = track_set_error(gt_tracks, tracks, max_dist=40)
        col_name = str(frame_rate) + 'Hz_' + algorithm
        error = pd.DataFrame.from_dict(error, orient='index', columns=[col_name])
        error = error.transpose()
        performance_measures = pd.concat((performance_measures, error))
    os.makedirs(os.path.join(results_file.split('/')[0], results_file.split('/')[1]), exist_ok=True)
    performance_measures.to_csv(results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute tracking performance measures')
    parser.add_argument('--dataset_dir', default='python_plus_octave/datasets_25_08_2020/dataset_1',
                        help='directory of the dataset')
    parser.add_argument('--results_file', default='error_results/dataset_performance_measures_results.csv',
                        help='file to save performance results')
    parser.add_argument('--save_vid', default=True, help='Flag')
    args = parser.parse_args()
    compute_errors(args.dataset_dir, args.results_file, args.save_vid)

    # DATASET_DIRS = ["python_plus_octave/datasets_25_08_2020/dataset_1",
    #                 "python_plus_octave/datasets_25_08_2020/dataset_2",
    #                 "python_plus_octave/datasets_25_08_2020/dataset_3",
    #                 "python_plus_octave/datasets_25_08_2020/dataset_4",
    #                 "python_plus_octave/datasets_25_08_2020/dataset_5",
    #                 "python_plus_octave/datasets_25_08_2020/dataset_6"]
    #
    # RESULTS_FILES = ["error_results/datasets_25_08_2020/dataset_1_performance_measures.csv",
    #                   "error_results/datasets_25_08_2020/dataset_2_performance_measures.csv",
    #                   "error_results/datasets_25_08_2020/dataset_3_performance_measures.csv",
    #                   "error_results/datasets_25_08_2020/dataset_4_performance_measures.csv",
    #                   "error_results/datasets_25_08_2020/dataset_5_performance_measures.csv",
    #                   "error_results/datasets_25_08_2020/dataset_6_performance_measures.csv"]

    # for i, dataset in enumerate(DATASET_DIRS):
    #     results = RESULTS_FILES[i]
    #     compute_errors(dataset, results)
