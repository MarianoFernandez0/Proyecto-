from evaluation import evaluation
from change_format_for_jpdaf import write_csv_for_jpdaf, save_secuence_as_jpgs_for_jpdaf
import tifffile
import os
from subprocess import call

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
    print(os.listdir(video_sequence_path))
    if len(os.listdir(video_sequence_path)) > 1:
        print("Must have only one sequence in directory")
        return
    print('Sequence name %s' % video_sequence)
    tiff = tifffile.TiffFile(video_sequences_path + '/' + video_sequence)
    detected = evaluation(tiff, include_mask=True)
    folder_directory = 'sequences_for_jpdaf/'
    save_secuence_as_jpgs_for_jpdaf(tiff, folder_directory)
    write_csv_for_jpdaf(detected, folder_directory)
    return


########################################################
#  START
########################################################

video_sequences_path = "./data_in/sequences"
main_jpdaf_implementation(video_sequences_path)
call("./execute_jpdaf.sh")