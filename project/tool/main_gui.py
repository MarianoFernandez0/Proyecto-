import os
from tool.src.tracking.tracking import Tracker, delete_tmp
import PySimpleGUI as sg
from tool.src.gui.gui import display_input_gui, save_detections_gui, save_tracks_gui, save_vid_gui


if __name__ == '__main__':
    event, values = display_input_gui()
    if event not in (sg.WIN_CLOSED, 'Cancel', 'Cancelar'):

        tracker = Tracker(params=values)

        detections_csv = save_detections_gui()
        tracker.detect(detections_file=detections_csv)

        tracks_csv = save_tracks_gui(os.path.dirname(detections_csv))
        tracks = tracker.track(detections_file=detections_csv, tracks_file=tracks_csv)

        tracks_video = save_vid_gui(tracks, os.path.dirname(tracks_csv))
        if tracks_video:
            tracker.save_vid(tracks_file=tracks_csv, video_file=tracks_video)
    delete_tmp()
