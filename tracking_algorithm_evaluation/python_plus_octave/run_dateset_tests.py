import os
import multiprocessing
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='datasets', help='Dataset directory')
args = parser.parse_args()

dataset_dir = args.dataset_dir
datasets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for dataset in datasets:
    print('processing dataset: ', dataset[-1])
    configs = os.listdir(os.path.join(dataset_dir, dataset, 'tracking_configs'))
    results = [os.path.splitext(result)[0] for result in os.listdir(os.path.join(dataset_dir, dataset,
                                                                                 'tracking_results'))]
    commands = ['python tracking_urbano.py ' + os.path.join(dataset_dir, dataset, 'tracking_configs', config)
                for config in configs if os.path.splitext(config)[0] not in results]
    pool = multiprocessing.Pool()
    pool.map(os.system, commands)