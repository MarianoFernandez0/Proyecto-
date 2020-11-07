from readlif.reader import LifFile
import numpy as np
from .tracking.tracking import Tracker


def infer_tracking(seq, fps):
    Tracker()


def run_for_lif(dir_lif, config_base):
    file = LifFile(dir_lif)
