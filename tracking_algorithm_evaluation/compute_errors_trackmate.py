from error_measures import track_set_error
import os
import pandas as pd

performance_measures = pd.DataFrame()

dataset_dir = 'Trackmate'
datasets_list = os.listdir(dataset_dir)

for dataset_folder in datasets_list:
    dataset_num = dataset_folder[7]
    dataset_frec = dataset_folder[9:11]
    gt_dir = dataset_dir + '/' + dataset_folder + '/' + 'dataset_' + dataset_num + '_' + dataset_frec + '.csv'
    print('gt_dir', gt_dir)
    gt_tracks = pd.read_csv(gt_dir)
    tracks_dir = (dataset_dir + '/' + dataset_folder + '/' + 'Spots in tracks statistics-dataset' + dataset_num + '-' +
                  dataset_frec + 'Hz.csv')
    print(tracks_dir)
    tracks_csv = pd.read_csv(tracks_dir)

    tracks_csv.rename(columns={'TRACK_ID': 'id',
                               'POSITION_X': 'x',
                               'POSITION_Y': 'y',
                               'FRAME': 'frame'}, inplace=True)
    tracks_csv = tracks_csv[tracks_csv['frame'] < tracks_csv['frame'].max()]

    gt_tracks = gt_tracks[gt_tracks['frame'] < tracks_csv['frame'].max()]
    gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
    gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)

    error = track_set_error(gt_tracks, tracks_csv, 40)
    error = pd.DataFrame.from_dict(error, orient='index', columns=['dataset_' + dataset_num + '_' +
                                                                   dataset_frec + 'Hz_trackmate'])
    error = error.transpose()
    performance_measures = pd.concat((performance_measures, error))
performance_measures.to_csv('performance_measures_trackmate.csv')
