import os
import PySimpleGUI as sg
from src.tracking.tracking import Tracker
from src.gui.gui import display_input_gui, drawing_vid_gui, progress_gui


def main():
    event, values = display_input_gui()
    if event not in (sg.WIN_CLOSED, 'Cancel', 'Cancelar'):
        tracker = Tracker(params=values)

        # detect
        window = progress_gui('Detecting...')
        tracker.detect()
        window.close()

        # track
        window = progress_gui('Tracking...')
        tracks = tracker.track()
        window.close()

        # save_vid
        window = drawing_vid_gui(tracks)
        tracker.save_vid()
        window.close()

        # who_measures
        window = progress_gui('Computing WHO measures...')
        tracker.who_measures()
        window.close()

        # who_classification
        window = progress_gui('Classifying...')
        tracker.who_classification()
        window.close()


if __name__ == '__main__':
    main()
