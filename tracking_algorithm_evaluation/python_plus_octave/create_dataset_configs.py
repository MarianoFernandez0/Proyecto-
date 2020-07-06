import os
import sys
import re
import json


def get_freq(file_name):
    idx1 = file_name.index('(')
    idx2 = file_name.index(')', idx1)
    freq = int(re.search(r'\d+', file_name[idx1:idx2]).group())
    return freq


datasets_dir = sys.argv[1]
default_config_dir = sys.argv[2]

att = ['NN', 'GNN', 'pdaf', 'jpdaf', 'ennjpdaf']

with open(default_config_dir, 'r') as f:
    config = json.load(f)

datasets_list = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
for dataset in datasets_list:
    data_files = os.listdir(os.path.join(datasets_dir, dataset, 'datasets/data_sequence'))
    if not os.path.isdir(os.path.join(datasets_dir, dataset, 'tracking_configs')):
        os.mkdir(os.path.join(datasets_dir, dataset, 'tracking_configs'))

    if not os.path.isdir(os.path.join(datasets_dir, dataset, 'tracking_results')):
        os.mkdir(os.path.join(datasets_dir, dataset, 'tracking_results'))

    for data_file in data_files:
        frequency = get_freq(data_file)
        mp4_dir = os.path.join(datasets_dir, dataset, 'datasets/video_sequence/mp4_output')
        mp4_file = [f for f in os.listdir(mp4_dir) if str(frequency) in f][0]
        tiff_dir = os.path.join(datasets_dir, dataset, 'datasets/video_sequence/tiff_output')
        tiff_file = [f for f in os.listdir(tiff_dir) if str(frequency) in f][0]

        config['Input']['DATAFILE_PATH'] = os.path.join(datasets_dir, dataset, 'datasets/data_sequence', data_file)
        config['Input']['VIDEOFILE_MP4_PATH'] = os.path.join(mp4_dir, mp4_file)
        config['Input']['VIDEOFILE_TIFF_PATH'] = os.path.join(tiff_dir, tiff_file)
        config['Input']['fps'] = frequency

        for i, _ in enumerate(att):
            config['Algorithm params']['detectionAlgorithm'] = i
            config_dir = os.path.join(datasets_dir, dataset, 'tracking_configs', dataset + '_' + str(frequency)
                                      + 'Hz_' + str(att[i]) + '.json')

            config['Output']['CSV_TRACKS_PATH'] = os.path.join(datasets_dir, dataset, 'tracking_results', dataset + '_'
                                                               + str(frequency) + 'Hz_' + str(att[i]) + '.csv')
            config['Output']['VIDEOFILE_OUT_PATH'] = os.path.join(datasets_dir, dataset, 'tracking_results', dataset +
                                                                  '_' + str(frequency) + 'Hz_' + str(att[i]) + '.csv')

            with open(config_dir, 'w') as f:
                json.dump(config, f)
