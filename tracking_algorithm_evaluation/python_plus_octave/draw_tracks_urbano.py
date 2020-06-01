from functions.draw_tracks import draw_tracks
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite as mp4_writer
import configparser
import sys


params = 'params.txt'

if len(sys.argv) > 1:
    params = sys.argv[1]

config = configparser.ConfigParser()
config.read(params)
config.sections()

video_file_tiff = config["Input"]["VIDEOFILE_TIFF_PATH"]
video_file_out = config["Output"]["VIDEOFILE_OUT_PATH"]
csv_tracks = config["Output"]["CSV_TRACKS_PATH"]

tiff = tifffile.TiffFile(video_file_tiff)
sequence = tiff.asarray()[:, :, :]
tracks_df = pd.read_csv(csv_tracks)
tracks = tracks_df.to_numpy()
tracks = tracks[tracks[:, 4] < 200]
tracks[:, 3] = np.zeros(tracks.shape[0])
sequence_tracks = draw_tracks(sequence, tracks)
mp4_writer(video_file_out, sequence_tracks, format='mp4', fps=5)

