from functions.evaluation import evaluation
from functions.change_format_for_jpdaf import write_csv_for_jpdaf, save_secuence_as_jpgs_for_jpdaf
import tifffile
import os
from draw_tracks import draw_tracks
from subprocess import call
import pandas as pd
from numpy import nan

def main_jpdaf_implementation(video_sequence_path):
    '''
    Main of the implementation.
    To run it, just call the "jpdaf_implementation.py arg1 arg2" from the command line.
        arguments (arg1, arg2..) are:
         the path for the sequence
         the path of the csv output
         the path for the images jpg
    '''

    video_sequence = os.listdir(video_sequence_path)[0]
    
    if len(os.listdir(video_sequence_path)) == 1:
        video_sequence = os.listdir(video_sequence_path)[0]
    elif video_sequence == os.listdir(video_sequence_path)[0][0] != ".":     
        video_sequence = os.listdir(video_sequence_path)[0]
    else:
        video_sequence = os.listdir(video_sequence_path)[1]        


    print(os.listdir(video_sequence_path))

    print('Sequence name %s' % video_sequence)
    tiff = tifffile.TiffFile(video_sequence_path + '/' + video_sequence)
    detected = evaluation(tiff, include_mask=True)

    folder_directory = 'sequences_for_jpdaf/'
    save_secuence_as_jpgs_for_jpdaf(tiff, folder_directory)
    write_csv_for_jpdaf(detected, folder_directory)
    return 0


########################################################
#  START
########################################################

video_sequences_path = "./data_in/sequences"
call("./execute_jpdaf.sh") if main_jpdaf_implementation(video_sequences_path) == 0 else print("error")
video_sequence = os.listdir(video_sequences_path)[0]
tiff = tifffile.TiffFile(video_sequences_path + '/' + video_sequence)
seq = tiff.asarray()
csv_output = pd.read_csv("./output/tracks.csv")
#print(csv_output.insert(3, "fluorescence", -1))
tracks_drawed = draw_tracks(seq, csv_output.to_numpy())
with tifffile.TiffWriter("./output/tracks_drawed.tiff", bigtiff=True) as tif:
    for frame in range(tracks_drawed.shape[0]):
        tif.save((tracks_drawed[frame]))
from imageio import mimwrite as mp4_writer
mp4_writer("./output/tracks_drawed.mp4", tracks_drawed, format="mp4", fps=5.)