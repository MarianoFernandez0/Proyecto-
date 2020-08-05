import pandas as pd
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os


def read_cvs(csv_path):
    return pd.read_csv(csv_path)


def build_cost_matrix(tracks_coord, detections_coord):
    return cdist(tracks_coord, detections_coord)


def add_fluorescence_to_tracks(detections, tracks, output_dir="output", type_measure='mgv'):
    """
    Function that assigns the fluorescence in the detections file to the particles in the track file.
    Args:
        detections: path to csv file with the detections for the sequence
        tracks: path to csv file with the output of the tracking algorithm
        output_dir: string with the path to save the csv file with fluorescence
        type_measure: str with ctcf or mgv
    """

    os.makedirs(output_dir, exist_ok=True)
    fluo_det_df = read_cvs(detections)
    tracks_df = read_cvs(tracks)
    tracks_df = tracks_df.assign(fluorescence=np.nan)

    measures_types = {'mgv': 'mean_gray_value',
                      'ctcf': 'ctcf'}

    for frame in list(dict.fromkeys(tracks_df['frame'])):
        tracks_in_frame = tracks_df[tracks_df['frame'] == frame]
        if not tracks_in_frame.empty:
            detections_in_frame = fluo_det_df[fluo_det_df['frame'] == frame]

            cost_matrix = build_cost_matrix(tracks_in_frame[['x', 'y']].to_numpy(),
                                            detections_in_frame[['x', 'y']].to_numpy())
            assigs = linear_sum_assignment(cost_matrix)

            ids_to_assig = (tracks_in_frame[['id']].to_numpy())[assigs[0]]
            values_to_assig = (detections_in_frame[measures_types[type_measure]].to_numpy())[assigs[1]]
            for i, id in enumerate(ids_to_assig):
                tracks_df.loc[np.logical_and(tracks_df['id'] == id[0], tracks_df['frame'] == frame), 'fluorescence'] = \
                values_to_assig[i]
    tracks_df.to_csv(output_dir + '/%s.csv' % (detections.split('/')[-1]).split(".")[0])


# testing
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--detections", help="Path to the detecctions file", required=True)
    parser.add_argument("--tracks", help="Path to csv with the tracking results", required=True)
    parser.add_argument("--outdir", default="output", help="Output folder")
    parser.add_argument("--fluo_type", default="mgv", help="Type of measurement used to measure the fluorescence")
    args = parser.parse_args()
    add_fluorescence_to_tracks(args.detections, args.tracks, args.outdir)
