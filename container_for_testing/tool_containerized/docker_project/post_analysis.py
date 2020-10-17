import numpy as np
import os


def make_comparission():
    base = '../data/datasets/'
    datasets = sorted(os.listdir(base))
    freq = sorted(os.listdir(os.path.join(base, datasets[0])))[0]
    print(os.path.join(base, datasets[0], freq, 'dataset_{}Hz_data_WHO.csv'.format(freq)))
    path_to_csv = os.path.join(base, datasets[0], freq, 'dataset_{}Hz_data_WHO.csv'.format(freq))
    data = np.genfromtxt(path_to_csv, dtype= float, delimiter=',', names=True)
    print(data.dtype.names)


if __name__ == "__main__":
    measures = {
        'vcl': [],
        'vsl': [],
        'vap_mean': []

    }
    make_comparission()

