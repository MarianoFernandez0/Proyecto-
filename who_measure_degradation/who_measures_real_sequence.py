from get_who_measures import get_casa_measures
import os
from tifffile import TiffFile
from PIL import Image
from PIL.TiffTags import TAGS


if __name__ == "__main__":

    indir = 'real_sequences'
    out_base = 'measures'
    seq_files = 'tiff_files'
    seq_files_path = os.path.join(indir, seq_files)
    sequences = os.listdir(seq_files_path)
    resolutions = {}
    fps = 30
    for seq in sequences:
        with TiffFile('real_sequences/tiff_files/10.tif') as img:
            tag = img.pages[0].tags['XResolution']
            # um/px
            res = tag.value[1] / tag.value[0]
            resolutions[(seq.split('.'))[0]] = res
    for sequence in resolutions.keys():
        res = resolutions[sequence]
        track_path = os.path.join(indir, sequence, 'tracks.csv')
        print(track_path)
        get_casa_measures(track_path, os.path.join(indir, sequence), res, fps)