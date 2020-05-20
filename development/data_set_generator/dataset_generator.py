#!/usr/bin/python
# coding: latin-1

from skimage.filters import gaussian
from skimage.draw import ellipse
from skimage.external.tifffile import TiffWriter
import numpy as np
import pandas as pd
import os
import math

HOUSING_PATH_SEQ_OUT = os.path.join("datasets", "video_sequence")
HOUSING_PATH_SEQ_DATA = os.path.join("datasets", "data_sequence")

def fetch_output(housing_path_seq_data=HOUSING_PATH_SEQ_DATA, housing_path_seq_out=HOUSING_PATH_SEQ_OUT):
    """
    This function takes the output files paths. If exists, does nothing, if not
    creates de directory

    Inputs:
        - housing_path_seq_data : A string with the path where is going
        to be saved the output data
        - housing_path_seq_out : A string with the path where is going
        to be saved the output sequences
    """
    if not os.path.isdir(housing_path_seq_data):
        os.makedirs(housing_path_seq_data)
    if not os.path.isdir(housing_path_seq_out):
        os.makedirs(housing_path_seq_out)
    return
