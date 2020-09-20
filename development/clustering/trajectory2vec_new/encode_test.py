import cv2
import tifffile
import numpy as np
from sklearn.cluster import KMeans
from trajectory_encoding import tracks_to_list, complete_trajectories, compute_feas, generate_behavior_sequences, \
    generate_normal_behavior_sequence, trajectory2Vec
from argparse import ArgumentParser

def encode_tracks(tracks, tif, video, encodes):
    trajectories = np.genfromtxt(tracks, delimiter=',', skip_header=True)  # id,x,y,fluorescence,frame
    # # print 'trajectories:\n', trajectories
    # trajectories_list = tracks_to_list(trajectories)
    # # print 'trajectories_list:\n', trajectories_list
    # trajectories_com = complete_trajectories(trajectories_list)
    # # print 'trajectories_com:\n', trajectories_com
    # trajectories_feas = compute_feas(trajectories_com)
    # # print 'trajectories_feas:\n', trajectories_feas
    # behavior_sequences = generate_behavior_sequences(trajectories_feas)
    # # print 'behavior_sequences:\n', behavior_sequences
    # behavior_sequences_normal = generate_normal_behavior_sequence(behavior_sequences)
    # # print 'behavior_sequences_normal:\n', behavior_sequences_normal
    # trajectories_vecs = trajectory2Vec(behavior_sequences_normal)
    # # print 'trajectories_vecs:\n', trajectories_vecs
    #
    # encodes_list = []
    # for tr in trajectories_vecs:
    #     encodes_list.append(tr[0][0])
    # encodes_array = np.array(encodes_list)
    # np.save(encodes, encodes_array)
    encodes_array = np.load(encodes + '.npy')
    print 'encodes shape', encodes_array.shape

    ids = np.unique(trajectories[:, 0])
    print 'ids', len(ids)
    frames = np.unique(trajectories[:, 4])
    print 'frames', len(frames)

    km = KMeans(n_clusters=4)
    km.fit(encodes_array)
    print km.labels_

    tif = tifffile.TiffFile(tif)
    sequence = tif.asarray()
    print 'sequence '
    print sequence.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video, fourcc, 4, (512, 512))

    colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
    img_old = np.zeros((512, 512, 3))
    for i, frame in enumerate(frames):
        frame_tracks = trajectories[trajectories[:, 4] == frame]
        # frame_img = cv2.cvtColor(sequence[i, :, :, :], cv2.COLOR_RGB2BGR)
        frame_img = cv2.cvtColor(sequence[int(frame), :, :], cv2.COLOR_GRAY2RGB)
        img = np.zeros((512, 512, 3)) + img_old
        frame_ids = np.unique(frame_tracks[:, 0])
        for i, track_id in enumerate(frame_ids):
            track = frame_tracks[frame_tracks[:, 0] == track_id]
            img_color = colors[int(km.labels_[np.argwhere(ids == track_id)])]
            cv2.circle(img, (int(track[:, 1]), int(track[:, 2])), 2, img_color)
        draw_img = frame_img.copy()
        draw_img[img != 0] = img[img != 0]
        out.write(draw_img.astype(np.uint8))
        img_old = img
    out.release()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--i', default=1, type=int)
    args = parser.parse_args()
    tracks_files = ['../input/dataset_1_15Hz_ennjpdaf/dataset_1_15Hz_ennjpdaf.csv',
                    '../input/dataset_2_15Hz_ennjpdaf/dataset_2_15Hz_ennjpdaf.csv',
                    '../input/dataset_3_15Hz_ennjpdaf/dataset_3_15Hz_ennjpdaf.csv',
                    '../input/dataset_4_15Hz_ennjpdaf/dataset_4_15Hz_ennjpdaf.csv',
                    '../input/dataset_5_15Hz_ennjpdaf/dataset_5_15Hz_ennjpdaf.csv',
                    '../input/dataset_6_15Hz_ennjpdaf/dataset_6_15Hz_ennjpdaf.csv',
                    '../input/1_026/tracks.csv',
                    '../input/good_video/tracks.csv']
    tif_files = ['../input/dataset_1_15Hz_ennjpdaf/dataset_1(15Hz).tif',
                 '../input/dataset_2_15Hz_ennjpdaf/dataset_2(15Hz).tif',
                 '../input/dataset_3_15Hz_ennjpdaf/dataset_3(15Hz).tif',
                 '../input/dataset_4_15Hz_ennjpdaf/dataset_4(15Hz).tif',
                 '../input/dataset_5_15Hz_ennjpdaf/dataset_5(15Hz).tif',
                 '../input/dataset_6_15Hz_ennjpdaf/dataset_6(15Hz).tif',
                 '../input/1_026/1 026.tif',
                 '../input/good_video/good-video.tif']
    video_files = ['../input/dataset_1_15Hz_ennjpdaf/video.avi',
                   '../input/dataset_2_15Hz_ennjpdaf/video.avi',
                   '../input/dataset_3_15Hz_ennjpdaf/video.avi',
                   '../input/dataset_4_15Hz_ennjpdaf/video.avi',
                   '../input/dataset_5_15Hz_ennjpdaf/video.avi',
                   '../input/dataset_6_15Hz_ennjpdaf/video.avi',
                   '../input/1_026/video.avi',
                   '../input/good_video/video.avi']
    encodes_files = ['../input/dataset_1_15Hz_ennjpdaf/encodes',
                     '../input/dataset_2_15Hz_ennjpdaf/encodes',
                     '../input/dataset_3_15Hz_ennjpdaf/encodes',
                     '../input/dataset_4_15Hz_ennjpdaf/encodes',
                     '../input/dataset_5_15Hz_ennjpdaf/encodes',
                     '../input/dataset_6_15Hz_ennjpdaf/encodes',
                     '../input/1_026/encodes',
                     '../input/good_video/encodes']

    # for i, trajs in enumerate(tracks_files):
    #     encode_tracks(tracks_files[i], tif_files[i], video_files[i], encodes_files[i])
    i = args.i
    print tracks_files[i], tif_files[i], video_files[i], encodes_files[i]
    encode_tracks(tracks_files[i], tif_files[i], video_files[i], encodes_files[i])