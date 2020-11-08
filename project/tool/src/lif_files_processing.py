from readlif.reader import LifFile
import numpy as np
import os
from .tracking.tracking import Tracker
from tifffile import TiffWriter
# from joblib import Parallel, delayed

def infer_tracking(sequence, config_base):
    fps = sequence.info['scale'][-1]
    res = 1/sequence.info['scale'][0]
    config_base['fps'] = fps
    config_base['um_per_px'] = res
    name = (sequence.info['name'].replace(',', '')).replace(' ', '_')
    path_out_tiff = os.path.join(config_base['video_input'], name + '.tif')
    config_base['video_input'] = path_out_tiff

    with TiffWriter(str(path_out_tiff), bigtiff=True) as tif:
        for t in range(sequence.dims[-1]):
            tif.save(np.array(sequence.get_frame(t=t)))

    tracker = Tracker(config_base)
    # detect
    tracker.detect()
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
    for seq in file.get_iter_image():
        infer_tracking((seq, config_base))

