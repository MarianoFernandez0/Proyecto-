import pandas as pd
import os
print('0', os.getcwd())
print('1', os.listdir(os.getcwd()))
print('2', os.listdir('python_plus_octave/datasets_19_08_2020/dataset_1/datasets/data_sequence'))
datasets = ["python_plus_octave/datasets_19_08_2020/dataset_1/datasets/data_sequence",
            "python_plus_octave/datasets_19_08_2020/dataset_2/datasets/data_sequence",
            "python_plus_octave/datasets_19_08_2020/dataset_3/datasets/data_sequence",
            "python_plus_octave/datasets_19_08_2020/dataset_4/datasets/data_sequence",
            "python_plus_octave/datasets_19_08_2020/dataset_5/datasets/data_sequence",
            "python_plus_octave/datasets_19_08_2020/dataset_6/datasets/data_sequence"]

csvs = ["(7Hz)_data.csv",
        "(15Hz)_data.csv",
        "(30Hz)_data.csv",
        "(40Hz)_data.csv"]

divs = [9, 4, 2, 2]

for n, dataset in enumerate(datasets):
    print(dataset)
    for i, csv in enumerate(csvs):
        name = 'dataset_' + str(n + 1) + csv
        print(name)
        print('list', os.listdir(dataset))
        gt = pd.read_csv(os.path.join(dataset, name))
        gt['frame'] = gt['frame']/divs[i]
        gt.to_csv(os.path.join(dataset, name))
