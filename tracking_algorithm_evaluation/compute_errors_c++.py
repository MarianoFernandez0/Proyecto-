from error_measures import track_set_error
import os
import pandas as pd

performance_measures = pd.DataFrame()

dataset_dir = 'tracks_c++'
datasets_list = os.listdir(dataset_dir)

for dataset in datasets_list:
    if len(dataset) == 18:
        dataset_num = dataset[8]
        dataset_frec = dataset[10:12]
        gt_dir = dataset_dir + '/' + 'dataset_' + dataset_num + '_' + dataset_frec + 'Hz_data.csv'
        print('Ground truth: ', gt_dir)
        gt_tracks = pd.read_csv(gt_dir)
        tracks_dir = (dataset_dir + '/' + 'dataset_' + dataset_num + '_' + dataset_frec + 'Hz.csv')
        print('Track: ', tracks_dir)
        tracks_csv = pd.read_csv(tracks_dir)
        tracks_csv.rename(columns={'track_id': 'id'}, inplace=True)
        tracks_csv = tracks_csv[tracks_csv['frame'] < tracks_csv['frame'].max()]

        gt_tracks = gt_tracks[gt_tracks['frame'] < tracks_csv['frame'].max()]
        gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
        gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)

        error = track_set_error(gt_tracks, tracks_csv, 40)
        error = pd.DataFrame.from_dict(error, orient='index', columns=['dataset_' + dataset_num + '_' +
                                                                       dataset_frec + 'Hz_c++'])
        error = error.transpose()
        performance_measures = pd.concat((performance_measures, error))
performance_measures.to_csv('performance_measures_c++.csv')
