from tool.src.tracking.tracking import Tracker, delete_tmp
from tool.src.error_measures.error_measures import track_set_error
from random_generator.generate_random_dataset import generate_config_file
from tool.src.who_measures.get_who_measures import get_casa_measures
import os
import json
import shutil
import argparse
import pandas as pd

def who_measures(track_path, fps):
    get_casa_measures(track_path, os.path.join(*track_path.split('/')[:-1]), 1, fps)


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
        shutil.move(os.path.join(path, 'video_sequence/mp4_output', video_file), os.path.join('../data/datasets/dataset_{}'.format(ds_number), f, video_file))
    shutil.rmtree(os.path.join(path))
    return '../data/datasets/dataset_{}'.format(ds_number)

def do_all_tracking_stuff(config):
    tracker = Tracker(params=config)
    tracker.detect()
    tracker.track()
    tracker.get_who_measures()
    tracker.who_classification()
    return tracker

def get_errors(tracker, folder_path):
    tracks_file = tracker.outdir + "/tracks/" + tracker.basename + '_tracks.csv'
    max_dist = (tracker.gv / (tracker.um_per_px * tracker.fps))
    gt_file = [gt_file for gt_file in os.listdir(folder_path) if 'who' not in gt_file.lower()]
    gt_file = os.path.join(folder_path, gt_file[0])

    gt_tracks = pd.read_csv(gt_file)
    tracks = pd.read_csv(tracks_file)
    gt_tracks.rename(columns={'id_particle': 'id'})
    errors = track_set_error(gt_tracks, tracks, max_dist)
    os.makedirs(os.path.join(folder_path, 'error_measures'), exist_ok=True)
    with open(os.path.join(folder_path, 'error_measures', 'error_measures.txt'), 'w') as f:
        json.dump(errors, f)

def organize_output(dataset_path):
    inference_path = '../data/output/'
    target_path = os.path.join(dataset_path, 'inference')
    os.makedirs(target_path, exist_ok=True)
    shutil.move(inference_path, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_tracker', default='../data/input/sequence', help='Path where to move the synthetic data')
    parser.add_argument('--config_file', default='../data/input/config/config_tracking.json', help='Path to the config file for the tracker')
    parser.add_argument('--loops', default=2, help='Number of datasets in which the tool is tested')
    parser.add_argument('--shared_folder', default='../data', help='Folder shared between the docker and host')
    args = parser.parse_args()
    for dataset in range(args.loops):
        generate_config_file()
        dataset_path = organize_datasets(dataset)
        freqs = os.listdir(dataset_path)
        for freq in freqs:
            # Organize the data to do the tracking
            freq_path = os.path.join(dataset_path, freq)
            video = [v for v in os.listdir(freq_path) if v.endswith('.mp4')][0]
            input_tracker_path = os.path.join(args.in_tracker, video)
            os.makedirs(os.path.join(*input_tracker_path.split(sep='/')[:-1]), exist_ok=True)
            shutil.move(os.path.join(freq_path, video), input_tracker_path)
            # Loading the config file and changing only the video path and fps
            with open(args.config_file, 'r') as file:
                config = json.load(file)
            config['video_input'] = input_tracker_path
            config['fps'] = freq
            # Make the inference
            tracker = do_all_tracking_stuff(config)
            get_errors(tracker, freq_path)
            os.remove(input_tracker_path)
            organize_output(os.path.join(dataset_path, freq))

    delete_tmp()





