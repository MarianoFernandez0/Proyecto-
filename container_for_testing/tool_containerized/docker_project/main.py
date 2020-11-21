from tool.src.tracking.tracking import Tracker
from tool.src.error_measures.error_measures import track_set_error
from tool.src.who_measures.get_who_measures import get_casa_measures
from random_generator.generate_random_dataset import generate_config_file
import os
import json
import time
import shutil
import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

TRACKING_ALGORITHMS = ['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF']


def who_measures(track_path, fps):
    get_casa_measures(track_path, os.path.join(*track_path.split('/')[:-1], 'who_measures.csv'), 1, fps)


def organize_datasets(ds_number, path='datasets'):
    datasets = os.listdir(os.path.join(path, 'data_sequence'))
    datasets = [f for f in datasets if f.endswith('.csv')]
    datasets.sort()
    freqs = [(f.split('Hz')[0]).split('_')[-1] for f in datasets]
    for i, f in enumerate(freqs):
        os.makedirs(os.path.join('../data/datasets/dataset_{}'.format(ds_number), f), exist_ok=True)
        shutil.move(os.path.join(path, 'data_sequence', datasets[i]), os.path.join('../data/datasets/dataset_{}'.format(ds_number), f, datasets[i]))
        who_measures(os.path.join('../data/datasets/dataset_{}'.format(ds_number), f, datasets[i]), float(f))
        video_file = str(datasets[i].split('_data.csv')[0]) + str('.mp4')
        # shutil.copy(os.path.join(path, 'video_sequence/mp4_output', video_file), os.path.join('../data/datasets/dataset_{}'.format(ds_number), 'video_sequence', video_file))
        shutil.move(os.path.join(path, 'video_sequence/mp4_output', video_file), os.path.join('../data/datasets/dataset_{}'.format(ds_number), f, video_file))
    shutil.rmtree(os.path.join(path))
    return '../data/datasets/dataset_{}'.format(ds_number)


def do_all_tracking_stuff(config):
    tracker = Tracker(params=config)
    print('Making detections...')
    tracker.detect()
    print('Making tracking...')
    tracker.track()
    print('Calculating WHO measures...')
    tracker.who_measures()
    print('Classifying tracks...')
    tracker.who_classification()
    print('saving video...')
    tracker.save_vid()
    return tracker


def get_errors(tracker, folder_path, resulsts_dict):
    tracks_file = os.path.join(tracker.out_dir, 'trajectories.csv')
    max_dist = (tracker.gv / (tracker.um_per_px * tracker.fps))
    gt_file = [gt_file for gt_file in os.listdir(folder_path) if 'who' not in gt_file.lower() and gt_file.endswith('csv')]
    gt_file = os.path.join(folder_path, gt_file[0])
    gt_tracks = pd.read_csv(gt_file)
    tracks = pd.read_csv(tracks_file)
    # gt_tracks.rename(columns={'id_particle': 'id'})
    errors = track_set_error(gt_tracks, tracks, max_dist)
    os.makedirs(os.path.join(folder_path, 'error_measures'), exist_ok=True)
    with open(os.path.join(folder_path, 'error_measures', 'error_measures.txt'), 'w') as f:
        json.dump(errors, f)
    for error_measure in errors:
        resulsts_dict[error_measure].append(errors[error_measure])


def organize_output(dataset_path):
    inference_path = '../data/output/'
    target_path = os.path.join(dataset_path, 'inference')
    os.makedirs(target_path, exist_ok=True)
    shutil.move(inference_path, target_path)


def update_measures(ds_num, freq, measures_freq, measures_freq_gt, mtt):
    path_gt = '../data/datasets/dataset_{}/{}/who_measures.csv'.format(ds_num, freq, freq)
    path_tracking = '../data/datasets/dataset_{}/{}/{}/inference/output/who_measures.csv'.format(ds_num, freq, mtt)
    who_gt = np.genfromtxt(path_gt, dtype=float, delimiter=',', names=True)
    who_tk = np.genfromtxt(path_tracking, dtype=float, delimiter=',', names=True)
    for k in map_keys:
        measures_freq[freq][mtt][k].append(np.nanmean(who_tk[map_keys[k]]))
        measures_freq_gt[freq][mtt][k].append(np.nanmean(who_gt[map_keys[k]]))
    # shutil.rmtree('../data/datasets/dataset_{}'.format(ds_num))
    return measures_freq, measures_freq_gt


def save_results(measures_freq, measures_freq_gt):
    out_path_1 = '../data/final_analysis_tracker.json'
    out_path_2 = '../data/final_analysis_gt.json'

    with open(out_path_1, 'w') as f:
        json.dump(measures_freq, f, indent=4)
    with open(out_path_2, 'w') as f:
        json.dump(measures_freq_gt, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_tracker', default='../data/input/sequence', help='Path where to move the synthetic data')
    parser.add_argument('--config_file', default='../data/input/config/config_tracking.json', help='Path to the config file for the tracker')
    parser.add_argument('--loops', default=6, type=int, help='Number of datasets in which the tool is tested')
    parser.add_argument('--shared_folder', default='../data', help='Folder shared between the docker and host')
    args = parser.parse_args()

    map_keys = {
        'vcl': 'vcl',
        'vsl': 'vsl',
        'vap': 'vap_mean',
        'alh': 'alh_mean',
        'lin': 'lin',
        'wob': 'wob',
        'str': 'str',
        'bcf': 'bcf_mean',
        'mad': 'mad',
        'fluo': 'fluo'
    }

    measures_freq = {}
    measures_freq_gt = {}

    timer = time.time()
    for dataset in range(args.loops):
        generate_config_file()
        dataset_path = organize_datasets(dataset)
        freqs = os.listdir(dataset_path)
        for freq in freqs:
            if freq not in list(measures_freq.keys()):
                measures_freq[freq] = {}
                measures_freq_gt[freq] = {}
            for algorithm in TRACKING_ALGORITHMS:
                if algorithm not in list(measures_freq[freq].keys()):
                    measures_freq[freq][algorithm] = {
                        'vcl': [],
                        'vsl': [],
                        'vap': [],
                        'alh': [],
                        'lin': [],
                        'wob': [],
                        'str': [],
                        'bcf': [],
                        'mad': [],
                        'fluo': [],
                        'alpha': [],
                        'beta': [],
                        'TP Tracks': [],
                        'FN Tracks': [],
                        'FP Tracks': [],
                        'JSC Tracks': [],
                        'RMSE': [],
                        'Min': [],
                        'Max': [],
                        'SD': [],
                        'TP Positions': [],
                        'FN Positions': [],
                        'FP Positions': [],
                        'JSC Positions': [],
                        'OSPA': []
                    }
                    measures_freq_gt[freq][algorithm] = {
                        'vcl': [],
                        'vsl': [],
                        'vap': [],
                        'alh': [],
                        'lin': [],
                        'wob': [],
                        'str': [],
                        'bcf': [],
                        'mad': [],
                        'fluo': []
                    }
                # Organize the data to do the tracking
                freq_path = os.path.join(dataset_path, freq)
                video = [v for v in os.listdir(freq_path) if v.endswith('.mp4')][0]
                input_tracker_path = os.path.join(args.in_tracker, video)
                os.makedirs(os.path.join(*input_tracker_path.split(sep='/')[:-1]), exist_ok=True)
                # Poner move para borrar el video, copy para no borrarlo
                shutil.copy(os.path.join(freq_path, video), input_tracker_path)
                # Loading the config file and changing only the video path and fps
                with open(args.config_file, 'r') as file:
                    config = json.load(file)
                config['video_input'] = input_tracker_path
                config['fps'] = freq
                config["mtt_algorithm"] = algorithm
                # Make the inference
                tracker = do_all_tracking_stuff(config)
                get_errors(tracker, freq_path, measures_freq[freq][algorithm])
                organize_output(os.path.join(dataset_path, freq, algorithm))
                measures_freq, measures_freq_gt = update_measures(dataset, freq, measures_freq, measures_freq_gt, algorithm)
                save_results(measures_freq, measures_freq_gt)
            os.remove(input_tracker_path)
            os.remove(os.path.join(freq_path, video))

    with open('../data/final_analysis_tracker.json', 'w') as file:
        json.dump(measures_freq, file, indent=4)

    with open('../data/final_analysis_gt.json', 'w') as file:
        json.dump(measures_freq_gt, file, indent=4)




