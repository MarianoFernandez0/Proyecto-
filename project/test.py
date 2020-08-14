import cv2
import pandas as pd
import tifffile
import numpy as np


tracks = pd.read_csv('output/tracks.csv')
detected = pd.read_csv('output/detections.csv')
tiff = tifffile.TiffFile('input/dataset_1(15Hz).tif')
sequence = tiff.asarray()
# --------------------------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
tracks_group_frame = tracks.groupby('frame')
dets_group_frame = detected.groupby('frame')
for frame in tracks['frame'].unique():
    if frame == max(tracks['frame'].unique()):
        continue
    img = cv2.cvtColor(sequence[frame, :, :], cv2.COLOR_BGR2RGB)
    print('frame', frame)
    tracks_frame = tracks_group_frame.get_group(frame)
    tracks_frame_group = tracks_frame.groupby('id')
    print('tracks_frame', tracks_frame)
    for tr, track_id in enumerate(tracks_frame['id'].unique()):
        track = tracks_frame_group.get_group(track_id)
        if np.isnan(track['fluorescence']).any():
            continue
        cv2.putText(img, 'Id: %d, F : %d' % (track_id, track['fluorescence']),
                    (int(track['x']) + 5, int(track['y']) + 5),
                    font,
                    fontScale,
                    (0, 0, 255),
                    1)
        cv2.circle(img, (int(track['x']) + 5, int(track['y'])), 1, (0, 0, 255))

    dets_frame = dets_group_frame.get_group(frame)
    print('dets_frame', dets_frame)

    for idx in dets_frame.index:
        if np.isnan(dets_frame.loc[idx, 'mean_gray_value']).any():
            continue
        cv2.circle(img, (int(dets_frame.loc[idx, 'y']), int(dets_frame.loc[idx, 'x'])), 1, (0, 255, 0))
        cv2.putText(img, 'F : %d' % (dets_frame.loc[idx, 'mean_gray_value']),
                    (int(dets_frame.loc[idx, 'y']) + 5, int(dets_frame.loc[idx, 'x']) + 25),
                    font,
                    fontScale,
                    (0, 255, 0),
                    1)
    cv2.imshow("tracks", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
