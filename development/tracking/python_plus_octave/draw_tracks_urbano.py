from draw_tracks import draw_tracks
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite as mp4_writer


video_file_tiff = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/real/9/9.tif'
video_file_mp4 = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/real/9/' \
                 'tracking/9_tracks.mp4'
tracks_file = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/real/9' \
              '/tracking/tracks.csv'

# video_file_tiff = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/dataset/' \
#                   'tiff_ouput/_noise_added_0_1.tiff'
# video_file_mp4 = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/dataset/' \
#                  'tracking/_noise_added_0_1_tracks.mp4'
# tracks_file = '/home/mariano/Projects/TDE/git/Proyecto-/development/tracking/python_plus_octave/data/dataset/' \
#               'tracking/tracks.csv'

tiff = tifffile.TiffFile(video_file_tiff)
sequence = tiff.asarray()[:, :, :]
tracks_df = pd.read_csv(tracks_file)
tracks = tracks_df.to_numpy()
tracks = tracks[tracks[:, 4] < 200]
tracks[:, 3] = np.zeros(tracks.shape[0])
sequence_tracks = draw_tracks(sequence, tracks)
mp4_writer(video_file_mp4, sequence_tracks, format='mp4', fps=5)

