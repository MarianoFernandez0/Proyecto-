import os
import json
import numpy as np
from argparse import ArgumentParser
from trajectory_encoding import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='../data/lif')
    args = parser.parse_args()

    data_dir = args.data_dir
    secuences = sorted(os.listdir(data_dir))
    secuences = [s for s in secuences if os.path.isdir(os.path.join(data_dir, s))]
    trajectories = []
    for secuence in secuences:
        with open(os.path.join(data_dir, secuence, 'config.txt'), 'r') as f:
            config = json.load(f)
        fps = config['fps']
        um_per_px = config['um_per_px']
        # id,x,y,fluorescence,frame
        tracks_array = np.genfromtxt(os.path.join(data_dir, secuence, 'trajectories.csv'),
                                     delimiter=',', skip_header=True)
        tracks_list = tracks_to_list(tracks_array, fps, um_per_px)
        trajectories += tracks_list

    print 'trajectories_list:\n', len(trajectories), trajectories[0][:3]
    trajectories_com = complete_trajectories(trajectories)
    print 'trajectories_com:\n', len(trajectories_com), trajectories_com[0][:3]
    trajectories_feas = compute_feas(trajectories_com)
    print 'trajectories_feas:\n', len(trajectories_feas), trajectories_feas[0][:3]
    behavior_sequences = generate_behavior_sequences(trajectories_feas, windowsize=10, offset=5)
    print 'behavior_sequences:\n', len(behavior_sequences), behavior_sequences[0][:3]
    behavior_sequences_normal = generate_normal_behavior_sequence(behavior_sequences)
    print 'behavior_sequences_normal:\n', len(behavior_sequences_normal), behavior_sequences_normal[0][:3]
    trajectories_vecs = trajectory2Vec(behavior_sequences_normal, size=20)
    print 'trajectories_vecs:\n', len(trajectories_vecs), trajectories_vecs[0]

    encodes_list = []
    for tr in trajectories_vecs:
        encodes_list.append(tr[0][0])
    encodes_array = np.array(encodes_list)
    np.save(os.path.join(data_dir, 'encodes'), encodes_array)
    print 'encodes shape', encodes_array.shape

    km = KMeans(n_clusters=4)
    km.fit(encodes_array)

    last_num_tracks = 0
    for secuence in secuences:
        sequence_vid = tifffile.TiffFile(os.path.join(data_dir, 'video.tif')).asarray()
        tracks_array = np.genfromtxt(os.path.join(data_dir, secuence, 'trajectories.csv'),
                                     delimiter=',', skip_header=True)
        num_tracks = len(np.unique(tracks_array[:, 0]))
        draw_labels(sequence_vid, tracks_array, km.labels_[last_num_tracks:num_tracks],
                    os.path.join(data_dir, secuence))
        last_num_tracks = num_tracks

    ids = np.unique(trajectories[:, 0])
    tracks_labels = pd.DataFrame(columns=['id', 'label'])
    tracks_labels['id'] = ids.astype(int)
    tracks_labels['label'] = km.labels_
    tracks_labels.to_csv(os.path.join(data_dir, 'id_labels.csv'), index=False)
    print tracks_labels.groupby('label').describe()['id']['count'].astype(int)

    plot_fluo(trajectories, tracks_labels, data_dir)
