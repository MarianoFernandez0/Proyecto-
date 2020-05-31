import tifffile
import numpy as np

def change_frame_rate(path_to_file, sub_sample):
    '''
    Function that given an input path, reads the corresponding tiff file and
    makes another tiffile sampling with period sub_sample

    Inputs:
        - path_to_file: String to the location of the tiff file
        - sub_sample: Integer value.
    '''

    tiff_in = tifffile.TiffFile(path_to_file)
    sequence_in = tiff_in.asarray()
    sequence_out = sequence_in[np.arange(sequence_in.shape[0]) % sub_sample == 0]
    out_path = path_to_file.split(".")[0] + "_subsampled_%d.tiff" % sub_sample
    with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
        for frame in range(sequence_out.shape[0]):
            tif.save((sequence_out[frame]))
    return