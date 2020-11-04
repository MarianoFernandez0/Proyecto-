import os
import argparse
import numpy as np
import tifffile
from sklearn.cluster import KMeans
from trajectory_encoding import get_encodes, draw_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/individual/9')  # id,x,y,fluorescence,frame
    args = parser.parse_args()

    data_dir = args.data_dir
    trajectories = np.genfromtxt(os.path.join(data_dir, 'trajectories.csv'), delimiter=',', skip_header=True)
    print 'trajectories'
    print trajectories.shape
    print trajectories[:5, :]
    trajectories_2 = trajectories.copy()
    trajectories_2[:, 0] = trajectories_2[:, 0] + 1000
    print 'trajectories_2'
    print trajectories_2.shape
    print trajectories_2[:5, :]
    trajectories_2 = np.concatenate((trajectories, trajectories_2))
    print 'trajectories_2'
    print trajectories_2.shape
    print trajectories_2[:5, :]
    print np.unique(trajectories[:, 0])
    # encodes = get_encodes(trajectories)
    # np.save(os.path.join('data/test_2', 'encodes'), encodes)
    encodes = np.load(os.path.join('data/test_2', 'encodes.npy'))
    print np.unique(trajectories_2[:, 0])
    # encodes_2 = get_encodes(trajectories_2)
    # np.save(os.path.join('data/test_2', 'encodes_x2'), encodes_2)
    encodes_2 = np.load(os.path.join('data/test_2', 'encodes_x2.npy'))

    print 'trajectories'
    print trajectories.shape
    print trajectories[trajectories[:, 0] == 3]
    print 'encodes'
    print len(encodes)
    print encodes[0]
    print encodes[1]

    print 'trajectories_2'
    print trajectories_2.shape
    print trajectories_2[trajectories_2[:, 0] == 3]
    print 'encodes_2'
    print len(encodes_2)
    print encodes_2[0]
    print encodes_2[1]

    print 'trajectories_x2'
    print trajectories_2[trajectories_2[:, 0] == 1003]
    print encodes_2[len(np.unique(trajectories[:, 0]))]
    print encodes_2[len(np.unique(trajectories[:, 0]))+1]

    sequence = tifffile.TiffFile(os.path.join(data_dir, 'video.tif')).asarray()
    km = KMeans(n_clusters=4)
    km.fit(encodes)
    # os.makedirs('data/test_2/video1')
    draw_labels(sequence, trajectories, km.labels_, 'data/test_2/video1')

    km = KMeans(n_clusters=4)
    km.fit(encodes_2)
    # os.makedirs('data/test_2/video2')
    draw_labels(sequence, trajectories_2, km.labels_, 'data/test_2/video2')