import os
import tifffile
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tool.src.vis.draw_tracks import draw_tracks
from imageio import mimwrite


def determine_cost_matrix(dets1, dets2, max_dist):
    matrix = np.zeros((dets1.shape[0], dets2.shape[0] + dets1.shape[0]))
    for i, det_i in enumerate(dets1):
        coords_i = det_i[:2]
        for j, det_j in enumerate(dets2):
            # print('coords_i', coords_i)
            coords_j = det_j[:2]
            matrix[i, j] = np.linalg.norm(coords_i - coords_j)
    matrix[:, dets2.shape[0]: dets2.shape[0] + dets1.shape[0]] = max_dist
    return matrix


def hungarian_tracking(dets, max_dist=40):
    frames = np.unique(dets[:, 2])
    frame_ids = np.array(range(1, dets[dets[:, 2] == frames[0]].shape[0] + 1, 1))
    ids_list = list(frame_ids)
    for i, frame in enumerate(frames[:-1]):
        # print('frame', frame)
        frame_dets = dets[dets[:, 2] == frame]
        next_frame_dets = dets[dets[:, 2] == frames[i + 1]]
        cost_matrix = determine_cost_matrix(frame_dets, next_frame_dets, max_dist)
        # print('     ids_list', ids_list)
        # print('     cost_matrix', cost_matrix.shape)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        #
        # print('     col_ind', col_ind, col_ind.shape)
        # print('     row_ind', row_ind, row_ind.shape)
        # print('     frame_ids', frame_ids, frame_ids.shape)

        next_frame_ids = np.zeros(next_frame_dets.shape[0], dtype=int)
        next_frame_ind = np.array(range(next_frame_dets.shape[0]))
        # print('     next_frame_ind', next_frame_ind, next_frame_ind.shape)

        next_opt_ind = col_ind[np.isin(col_ind, next_frame_ind)]
        opt_ind = row_ind[np.isin(col_ind, next_frame_ind)]
        # print('     next_opt_ind', next_opt_ind, next_opt_ind.shape)
        # print('     opt_ind', opt_ind, opt_ind.shape)
        #
        # print('         cost_matrix[row_ind, col_ind]', cost_matrix[row_ind, col_ind].max())
        # print('         cost_matrix[opt_ind, next_opt_ind]', cost_matrix[opt_ind, next_opt_ind].max())
        new_ind = next_frame_ind[~np.isin(next_frame_ind, col_ind)]
        # print('     new_ind', new_ind, new_ind.shape)

        if len(new_ind) > 0:
            # print('np.array(range(1, len(new_ind)))', np.array(range(1, len(new_ind) + 1)))
            next_frame_ids[new_ind] = np.max(frame_ids) + np.array(range(1, len(new_ind) + 1))
        # print('     np.max(frame_ids) + np.array(range(len(new_ind)))',  np.max(frame_ids) + np.array(range(len(new_ind))))

        next_frame_ids[next_opt_ind] = frame_ids[opt_ind]
        # print('     next_frame_ids', next_frame_ids)

        frame_ids = next_frame_ids
        ids_list += list(frame_ids)
    tracks = dets.copy()
    ids_list = np.array(ids_list, dtype=int)[:, np.newaxis]
    print(tracks.shape, ids_list.shape)
    tracks = np.concatenate((dets, ids_list), axis=1)
    return tracks


if __name__ == '__main__':
    file = '../../output/dataset_1(15Hz)_old_params/detections.csv'
    # print(os.listdir(file))
    detections = pd.read_csv(file, index_col=False, usecols=['x', 'y', 'frame', 'mean_gray_value'])
    print(detections)
    trajectories = hungarian_tracking(detections.to_numpy())
    trajectories_cols = list(detections.columns) + ['id']
    print(trajectories_cols)
    trajectories = pd.DataFrame(data=trajectories, columns=trajectories_cols)
    print(trajectories)
    tiff = tifffile.TiffFile('../../input/dataset_1(15Hz).tif')
    sequence = tiff.asarray()
    tracks_data = trajectories[['id', 'y', 'x', 'mean_gray_value', 'frame']]
    tracks_data.rename(columns={'id': 'track_id', 'mean_gray_value': 'fluorescence'}, inplace=True)
    print(tracks_data)
    vid = draw_tracks(sequence, tracks_data.to_numpy())
    mimwrite('vid.mp4', vid)
