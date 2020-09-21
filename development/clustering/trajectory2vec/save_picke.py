import pandas as pd
import cPickle
import os


# tracks = pd.read_csv('real_data/tracks.csv')
tracks = pd.read_csv('../input/dataset_5_15Hz_ennjpdaf/dataset_5_15Hz_ennjpdaf.csv')
tracks_dict = {}
for i in tracks.index:
    if tracks.loc[i, 'id'] in tracks_dict.keys():
        tracks_dict[tracks.loc[i, 'id']].append([tracks.loc[i, 'frame'], tracks.loc[i, 'x'], tracks.loc[i, 'y']])
    else:
        tracks_dict[tracks.loc[i, 'id']] = [[tracks.loc[i, 'frame'], tracks.loc[i, 'x'], tracks.loc[i, 'y']]]
print tracks_dict

tracks_list = []
for k in tracks_dict.keys():
    tracks_list.append(tracks_dict[k])
print tracks_list

cPickle.dump(tracks_list, open('../input/dataset_5_15Hz_ennjpdaf/pickles/dataset_5_15Hz_ennjpdaf.p', 'w'))

