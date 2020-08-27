import os
import re
import json
import argparse


def get_freq(file_name):
    idx1 = file_name.index('(')
    idx2 = file_name.index(')', idx1)
    freq = int(re.search(r'\d+', file_name[idx1:idx2]).group())
    return freq


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='datasets_25_08_2020', help='Dataset directory')
    parser.add_argument('--default_config_dir', default='datasets_25_08_2020/config.json', help='Base config directory')

    args = parser.parse_args()
    datasets_dir = args.dataset_dir
    default_config_dir = args.default_config_dir

    att = ['NN', 'GNN', 'pdaf', 'jpdaf', 'ennjpdaf']

    with open(default_config_dir, 'r') as f:
        config = json.load(f)

    datasets_list = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    for dataset in datasets_list:
        data_files = os.listdir(os.path.join(datasets_dir, dataset, 'datasets/data_sequence'))

        os.makedirs(os.path.join(datasets_dir, dataset, 'tracking_configs'), exist_ok=True)
        os.makedirs(os.path.join(datasets_dir, dataset, 'detection_results'), exist_ok=True)
        os.makedirs(os.path.join(datasets_dir, dataset, 'tracking_results'), exist_ok=True)

        for data_file in data_files:
            frequency = get_freq(data_file)
            mp4_dir = os.path.join(datasets_dir, dataset, 'datasets/video_sequence/mp4_output')
            mp4_file = [f for f in os.listdir(mp4_dir)
                        if str(frequency) in f and 'noise' not in f and 'segmented' not in f][0]
            tiff_dir = os.path.join(datasets_dir, dataset, 'datasets/video_sequence/tiff_output')
            tiff_file = [f for f in os.listdir(tiff_dir)
                         if str(frequency) in f and 'noise' not in f and 'segmented' not in f][0]

            config['Input']['tif_video_input'] = os.path.join(tiff_dir, tiff_file)
            config['Input']['fps'] = frequency

            for i, _ in enumerate(att):
                config['Algorithm params']['mtt_algorithm'] = i + 1
                config_dir = os.path.join(datasets_dir, dataset, 'tracking_configs',
                                          dataset + '_' + str(frequency) + 'Hz_' + str(att[i]) + '.json')

                config['Output']['input_video'] = os.path.join(mp4_dir, mp4_file)
                config['Output']['detections_csv'] = os.path.join(datasets_dir, dataset, 'detection_results',
                                                                  dataset + '_' + str(frequency) + 'Hz_'
                                                                  + str(att[i]) + '.csv')
                config['Output']['tracks_csv'] = os.path.join(datasets_dir, dataset, 'tracking_results',
                                                              dataset + '_' + str(frequency) + 'Hz_'
                                                              + str(att[i]) + '.csv')
                os.makedirs(os.path.join(datasets_dir, dataset, 'tracking_results/videos'), exist_ok=True)
                config['Output']['tracks_video'] = os.path.join(datasets_dir, dataset, 'tracking_results/videos',
                                                                dataset + '_' + str(frequency) + 'Hz_'
                                                                + str(att[i]) + '.mp4')

                with open(config_dir, 'w') as f:
                    json.dump(config, f)
