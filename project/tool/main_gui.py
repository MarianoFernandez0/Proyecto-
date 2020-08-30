import json
import os
import sys
import shutil
from tool.src.tracking.tracking import Tracker

import PySimpleGUI as sg
from tool.src.gui.gui import display_input_gui, display_results_gui

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
    os.environ["OCTAVE_KERNEL_JSON"] = os.path.join(application_path, 'octave_kernel/kernel.json')
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
import oct2py

os.makedirs(os.path.join(application_path, 'tmp'), exist_ok=True)
octave = oct2py.Oct2Py(temp_dir=os.path.join(application_path, 'tmp'))
print('application_path', application_path)
# /home/mariano/Projects/TDE/git/Proyecto-/project/tool/src/SpermTrackingProject
print(os.path.join(application_path, 'src/SpermTrackingProject'))
octave.addpath(os.path.join(application_path, 'src/SpermTrackingProject'))
octave.addpath(os.path.join(application_path, 'src/oct2py'))

################################################################################################################
################################################################################################################

if __name__ == '__main__':
    event, values = display_input_gui()
    if event not in (sg.WIN_CLOSED, 'Cancel', 'Cancelar'):
        output_folder = values['output']

        detections_csv = os.path.join(output_folder, 'detections.csv')
        tracks_csv = os.path.join(output_folder, 'tracks.csv')
        tracks_video = os.path.join(output_folder, 'tracks.mp4')

        tracker = Tracker(params=values, octave_interpreter=octave)
        tracker.detect(detections_file=detections_csv)
        tracks = tracker.track(detections_file=detections_csv, tracks_file=tracks_csv)
        tracker.save_vid(tracks_file=tracks_csv, video_file=tracks_video)

        display_results_gui(tracks)

# delete temporal folder
shutil.rmtree(os.path.join(application_path, 'tmp'))
