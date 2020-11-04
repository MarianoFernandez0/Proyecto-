import cv2


def draw_dets(sequence, detections):
    for num_frame in range(sequence.shape[0]):
        frame_dets = detections[detections['frame'] == num_frame].to_numpy()
        for det in frame_dets:
            cv2.circle(sequence[num_frame], (int(det[2]), int(det[1])), 5, (0, 255, 0))
    return sequence
