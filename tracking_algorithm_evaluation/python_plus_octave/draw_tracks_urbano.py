from functions.draw_tracks import draw_tracks
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite as mp4_writer
import json
import sys


if len(sys.argv) > 1:
    params = sys.argv[1]

with open(params, 'r') as f:
    config = json.load(f)

video_file_tiff = config["Input"]["VIDEOFILE_TIFF_PATH"]
video_file_out = config["Output"]["VIDEOFILE_OUT_PATH"]
csv_tracks = config["Output"]["CSV_TRACKS_PATH"]
tiff = tifffile.TiffFile(video_file_tiff)
sequence = tiff.asarray()[:, :, :]
tracks_df = pd.read_csv(csv_tracks)
print(tracks_df.head())
# tracks = tracks_df.to_numpy()[:, 1:]
tracks = tracks_df.to_numpy()
print(tracks.shape)
tracks[:, 3] = np.zeros(tracks.shape[0])
#tracks = tracks[:, [0, 2, 1, 3, 4]]
tracks = tracks[tracks[:, 4] < tracks[:, 4].max()]
print(video_file_out)
sequence_tracks = draw_tracks(sequence, tracks)
mp4_writer(video_file_out, sequence_tracks, format='mp4', fps=5)

