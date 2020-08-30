import os
import sys
import shutil
from tool.src.tracking.tracking import Tracker

import PySimpleGUI as sg
from tool.src.gui.gui import display_input_gui, save_detections_gui, save_tracks_gui, save_vid_gui

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
    os.environ["OCTAVE_KERNEL_JSON"] = os.path.join(application_path, 'octave_kernel/kernel.json')
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
import oct2py

os.makedirs(os.path.join(application_path, 'tmp'), exist_ok=True)
octave = oct2py.Oct2Py(temp_dir=os.path.join(application_path, 'tmp'))
octave.addpath(os.path.join(application_path, 'src/SpermTrackingProject'))
octave.addpath(os.path.join(application_path, 'src/oct2py'))

################################################################################################################
################################################################################################################

if __name__ == '__main__':
    event, values = display_input_gui()
    if event not in (sg.WIN_CLOSED, 'Cancel', 'Cancelar'):

        tracker = Tracker(params=values, octave_interpreter=octave)

        detections_csv = save_detections_gui()
        tracker.detect(detections_file=detections_csv)

        tracks_csv = save_tracks_gui(os.path.dirname(detections_csv))
        tracks = tracker.track(detections_file=detections_csv, tracks_file=tracks_csv)

        tracks_video = save_vid_gui(tracks, os.path.dirname(tracks_csv))
        if tracks_video:
            tracker.save_vid(tracks_file=tracks_csv, video_file=tracks_video)

# delete temporal folder
shutil.rmtree(os.path.join(application_path, 'tmp'))
