import pandas as pd
import numpy as np
from src.draw_tracks import draw_tracks
from imageio import mimwrite, mimread

sequence_list = mimread('input/1472 semen-00.avi')
sequence = np.array(sequence_list)

tracks = pd.read_csv('output/campo_claro/tracks.csv', index_col=False)
tracks_array = tracks.to_numpy()
tracks_array[np.isnan(tracks_array)] = 0
tracks_array = tracks_array[tracks_array[:, 4] < tracks_array[:, 4].max()]
sequence_tracks = draw_tracks(sequence, tracks_array, text=False)
mimwrite('output/campo_claro/tracks.mp4', sequence_tracks, format='mp4', fps=10)