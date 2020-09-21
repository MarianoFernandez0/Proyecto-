import os
import numpy as np
from argparse import ArgumentParser
from trajectory_encoding import get_encodes, draw_labels
from sklearn.cluster import KMeans
import tifffile
import cv2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='../data/real_data')
    args = parser.parse_args()

    sequences = [l for l in os.listdir(args.data) if '.' not in l]
    all_tracks = []
    in_videos = []
    out_videos = []
    for i, seq in enumerate(sequences):
        # id,x,y,fluorescence,frame
        tracks = np.genfromtxt(os.path.join(args.data, seq, 'tracks.csv'), delimiter=',', skip_header=True)
        tracks = np.concatenate((tracks, np.ones((tracks.shape[0], 1))*i), axis=1)
        tracks[:, 0] += i*1000
        all_tracks.append(tracks)

        tif = tifffile.TiffFile(os.path.join(args.data, seq, 'video.tif'))
        in_video = tif.asarray()
        in_videos.append(in_video)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(os.path.join(args.data, seq, 'out_video.avi'), fourcc, 4, (512, 512))
        out_videos.append(out_video)

    all_tracks = np.concatenate(all_tracks, axis=0)

    # encodes = get_encodes(all_tracks[:, :-1])
    # np.save('../data/real_data/encodes', encodes)
    encodes = np.load(os.path.join(args.data, 'encodes.npy'))

    km = KMeans(n_clusters=4)
    km.fit(encodes)

    ids = np.unique(all_tracks[:, 0])
    frames = np.unique(all_tracks[:, 4])

    for k, in_video in enumerate(in_videos):
        tracks = all_tracks[all_tracks[:, 5] == k]
        track_labels = km.labels_[ids//1000 == k]
        draw_labels(in_video, tracks, track_labels, out_videos[k])