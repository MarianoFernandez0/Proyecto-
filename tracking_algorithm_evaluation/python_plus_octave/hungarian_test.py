import os
import multiprocessing
import argparse
import json
from tool.src.tracking.tracking import Tracker, delete_tmp
from tool.src.tracking.hungarian import hungarian_tracking
import pandas as pd


def do_tracking_hungarian(config):
    print('-----------------------------------------------------------------------------------------------------------')
    tracker = Tracker(params=config)
    # tracker.detect(detections_file=config['detections_csv'])
    detections = pd.read_csv(config['detections_csv'], index_col=False, usecols=['x', 'y', 'frame', 'mean_gray_value'])
    print('detections\n', detections)
    trajectories = hungarian_tracking(detections.to_numpy())
    print(trajectories)
    trajectories_cols = list(detections.columns) + ['id']
    trajectories = pd.DataFrame(data=trajectories, columns=trajectories_cols)
    tracks_data = trajectories[['id', 'y', 'x', 'mean_gray_value', 'frame']]
    tracks_data.rename(columns={'mean_gray_value': 'fluorescence'}, inplace=True)
    print('tracks_data\n', tracks_data)
    print(config['tracks_csv'])

    tracks_csv = config['tracks_csv'].split(sep='/')
    tracks_csv[5] = 'hungarian_results'
    tracks_csv = os.path.join(*tracks_csv)
    os.makedirs(os.path.dirname(tracks_csv), exist_ok=True)
    tracks_data.to_csv(tracks_csv, index=False)

    tracks_video = config['tracks_video'].split(sep='/')
    tracks_video[5] = 'hungarian_results'
    tracks_video = os.path.join(*tracks_video)

    os.makedirs(os.path.dirname(tracks_video), exist_ok=True)
    tracker.save_vid(tracks_file=tracks_csv, video_file=tracks_video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../../Datasets/dataset_29_8_2020', help='Dataset directory')
    args = parser.parse_args()

    datasets = [d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]

    for dataset in datasets:
        print('processing dataset: ', dataset[-1])
        configs = os.listdir(os.path.join(args.dataset_dir, dataset, 'tracking_configs'))
        results = [os.path.splitext(result)[0] for result in os.listdir(os.path.join(args.dataset_dir, dataset,
                                                                                     'tracking_results'))]
        configs_unique = [os.path.join(args.dataset_dir, dataset, 'tracking_configs', config) for config in configs
                          if '_NN' in config]
        print('{} tests to run: '.format(len(configs_unique)), sorted(configs_unique))
        configs_list = []
        for config in configs_unique:
            with open(config, 'r') as f:
                configs_list.append(json.load(f))
        if len(configs_list) > 0:
            for config in configs_list:
                config['detection_algorithm'] = 'Fluorescence'
                do_tracking_hungarian(config)

            # pool = multiprocessing.Pool(7)
            # pool.map(do_tracking, configs_list)
    delete_tmp()
