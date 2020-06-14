from error_measures import track_set_error
import os
import pandas as pd

performance_measures = pd.DataFrame()

dataset_dir = 'python_plus_octave/data'
frame_rate_list = os.listdir(dataset_dir)

for frame_rate in frame_rate_list:
    if 'Hz' in frame_rate:
        gt_dir = dataset_dir+'/'+frame_rate+'/'+'dataset_'+frame_rate[8]+'_data.csv'
        gt_tracks = pd.read_csv(gt_dir)
        print('gt_dir', gt_dir)
        algoritmns = os.listdir(dataset_dir+'/'+frame_rate)
        for algoritmn in algoritmns:
            if 'tracking' in algoritmn:
                tracks_dir = dataset_dir+'/'+frame_rate+'/'+algoritmn+'/dataset_'+frame_rate[8]+'/'+'tracks.csv'
                tracks_csv = pd.read_csv(tracks_dir)
                print(tracks_dir)
                tracks_csv = tracks_csv[tracks_csv['frame'] < tracks_csv['frame'].max()]
                gt_tracks = gt_tracks[gt_tracks['frame'] < tracks_csv['frame'].max()]

                gt_tracks = gt_tracks[gt_tracks['frame'] > 1]

                gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)
                error = track_set_error(gt_tracks, tracks_csv, 40)
                error = pd.DataFrame.from_dict(error, orient='index', columns=[frame_rate+'_'+algoritmn])
                error = error.transpose()
                performance_measures = pd.concat((performance_measures, error))
performance_measures.to_csv('performance_measures_octave.csv')
