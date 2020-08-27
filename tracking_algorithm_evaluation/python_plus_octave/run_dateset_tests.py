import os
import multiprocessing
import argparse
import json
from tool.main import tracking_urbano, TrackingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='datasets_25_08_2020', help='Dataset directory')
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
        args_list = []
        for config in configs_unique:
            with open(config, 'r') as f:
                args_list.append(TrackingParams(json.load(f)))
        if len(args_list) > 0:
            pool = multiprocessing.Pool()
            pool.map(tracking_urbano, args_list)
