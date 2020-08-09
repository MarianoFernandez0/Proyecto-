from cv2 import line, putText, FONT_HERSHEY_SIMPLEX
import numpy as np


def draw_tracks(sequence: np.array, tracks_data: np.array) -> np.array:
    '''
    Function that draws the tracks given the sequence and the array of tracks

    Input:
        - squence: array of shape (N,H,W,C)
            N: number of frames
            H: Height
            W: Width
            C: Channel (empty if grayscale)

        - tracks_data: array where:
                            tracks_data[0] track_id
                            tracks_data[1] x
                            tracks_data[2] y
                            tracks_data[3] fluorescence
                            tracks_data[4] frame

    Ouput:
        - seq_out: array of shape (N,H,W,C)
    '''
    frames, H, W = sequence.shape[0:3]
    seq_out = sequence.copy()

    if len(sequence.shape) < 4:
        seq_out = np.zeros((frames, H, W, 3))
        seq_out[:, :, :, 0] = sequence.copy()

    tracks_id = (np.unique(tracks_data[:, 0])).tolist()

    # Values for the cv2 writer in images
    font = FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    lineType = 1

    # Make a list of length total number of frames and put every track in the
    # tracks data array
    tracks = []
    for track_id in tracks_id:
        data_track = tracks_data[tracks_data[:, 0] == track_id]
        coord = np.ones((frames, 4)) * -1
        #print(np.uint(data_track[:, 4]))
        #print(coord.shape)
        coord[np.uint(data_track[:, 4])] = data_track[:, 0:4]
        tracks.append(coord)

    # Color for every track
    colors = np.random.uniform(0, 255, (len(tracks_id), 3))

    # Draws every tracks, and delete previous tracks frames so the finals frame are no
    # dense drawn
    for frame in range(frames):
        start = 1
        if frame > 5:
            start = frame - 5
        for i in range(start, frame + 1):

            for tr, track in enumerate(tracks):
                if track[i, 1] > 0 and track[i - 1, 1] > 0:
                    line(seq_out[frame], (int(track[i - 1, 1]), int(track[i - 1, 2])),
                             (int(track[i, 1]), int(track[i, 2])), colors[tr])

                    fontColor = colors[tr]
                    coord = (int(track[frame, 1]) + 5, int(track[frame, 2]) + 5)
                    # print('f', frame, (track[frame, 0], track[frame, 3]))
                    putText(seq_out[frame], 'Id: %d, F : %d' % (track[frame, 0], track[frame, 3]),
                                coord,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

    return np.uint8(seq_out)
