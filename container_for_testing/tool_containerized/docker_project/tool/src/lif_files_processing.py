from readlif.reader import LifFile
import numpy as np
import os
from .tracking.tracking import Tracker
from tifffile import TiffWriter
from joblib import Parallel, delayed


def infer_tracking(sequence, config_base):
    fps = sequence.info['scale'][-1]
    res = 1/sequence.info['scale'][0]
    config_base['fps'] = fps
    config_base['um_per_px'] = res
    name = (sequence.info['name'].replace(',', '')).replace(' ', '_')
    path_out_tiff = os.path.join(config_base['video_input'], name + '.tif')
    config_base['video_input'] = path_out_tiff
    config_base['out_dir'] = os.path.join(config_base['out_dir'], name)

    os.makedirs(config_base['out_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(path_out_tiff), exist_ok=True)
    if not os.path.isfile(str(path_out_tiff)):
        with TiffWriter(str(path_out_tiff), bigtiff=True) as tif:
            for t in range(sequence.dims[-1]):
                tif.save(np.array(sequence.get_frame(t=t)))

    if res > 0.6:
        return

    tracker = Tracker(config_base)
    # detect
    detections = tracker.detect()
    num_dets = detections[detections['frame'] == detections['frame'].unique()[0]].shape[0]
    print('dets', num_dets)
    if num_dets > 75:
        return

    # track
    tracks = tracker.track()
    # save_vid
    tracker.save_vid()
    # who_measures
    tracker.who_measures()
    # who_classification
    tracker.who_classification()


def run_for_lif(dir_lif, config_base):
    file = LifFile(dir_lif)
    # for seq in file.get_iter_image():
    #     infer_tracking(seq, config_base.copy())
    Parallel(n_jobs=7)(delayed(infer_tracking)(seq, config_base.copy()) for seq in file.get_iter_image())

