import os
import multiprocessing
import argparse
import json
from tool.src.tracking.tracking import Tracker, delete_tmp


def do_tracking(config):
    tracker = Tracker(params=config)
    tracker.detect(detections_file=config['detections_csv'])
    tracker.track(detections_file=config['detections_csv'], tracks_file=config['tracks_csv'])
    tracker.save_vid(tracks_file=config['tracks_csv'], video_file=config['tracks_video'])
    delete_tmp()


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
        configs_unique = [os.path.join(args.dataset_dir, dataset, 'tracking_configs', config)
                          for config in configs if os.path.splitext(config)[0] not in results]
        print('{} tests to run: '.format(len(configs_unique)), sorted(configs_unique))
        configs_list = []
        for config in configs_unique:
            with open(config, 'r') as f:
                configs_list.append(json.load(f))
        if len(configs_list) > 0:
            pool = multiprocessing.Pool()
            pool.map(do_tracking, configs_list)
