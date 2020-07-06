import os
import sys
import multiprocessing

dataset_dir = sys.argv[1]
datasets = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for dataset in datasets:
    print(dataset)
    configs = os.listdir(os.path.join(dataset_dir, dataset, 'tracking_configs'))
    commands = ['python tracking_urbano.py ' + os.path.join(dataset_dir, dataset, 'tracking_configs', config)
                for config in configs]
    pool = multiprocessing.Pool()
    pool.map(os.system, commands)
