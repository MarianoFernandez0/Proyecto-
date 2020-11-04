import os
import sys
import tifffile
import tempfile
import numpy as np
import pandas as pd
from imageio import mimwrite, mimread
from src.vis.draw_tracks import draw_tracks
from src.vis.detections import draw_dets
from src.detection.evaluation import evaluation
from src.detection.gray_detection import gray_evaluation
from src.fluorescence.add_fluorescence import add_fluorescence_to_tracks
from src.who_measures.get_who_measures import get_casa_measures
from src.classification.classification_WHO import classification
import cv2

if getattr(sys, 'frozen', False):  # if running from executable
    application_path = sys._MEIPASS
    TOOL_PATH = application_path.split(sep='/')
    TOOL_PATH[0] = '/'
    os.environ["OCTAVE_KERNEL_JSON"] = os.path.join('/', *TOOL_PATH, 'octave_kernel/kernel.json')
else:  # if running from python script
    application_path = os.path.dirname(os.path.abspath(__file__))
    TOOL_PATH = application_path.split(sep='/')
    TOOL_PATH.pop(-1)
    TOOL_PATH[0] = '/'
import oct2py
tmp = tempfile.TemporaryDirectory()
octave = oct2py.Oct2Py(temp_dir=tmp.name)
octave.addpath(os.path.join(*TOOL_PATH, 'SpermTrackingProject'))
octave.addpath(os.path.join(*TOOL_PATH, 'oct2py'))


class Tracker:
    """
    Attributes:
        fps (int): Frame frequency.
        um_per_px (float): Scale of the image.
        detection_algorithm (int): Detection algorithm.
                                    0 = MatLab implementation
                                    1 = Python implementation
        mtt_algorithm (int): Multi-Target Tracking algorithm.
                                1 = NN
                                2 = GNN
                                3 = PDAF
                                4 = JPDAF
                                5 = ENN-JPDAF
                                6 = Iterated Multi-assignment
        mtt_algorithm (int): Multi-Target Tracking algorithm.
        PG (float): Prob. that a detected target falls in validation gate
        PD (float): Prob. of detection
        gv (float): Velocity Gate (um/s)
        sequence (np.ndarray): Video sequence, shape (N, H, W).

    """
    def __init__(self, params):
        self.octave = octave
        self.fps = int(params['fps'])

        if isinstance(params['um_per_px'], (int, float)) or params['um_per_px'].replace('.', '', 1).isdigit():
            self.um_per_px = float(params['um_per_px'])
        else:
            self.um_per_px = None

        self.detection_algorithm = params['detection_algorithm']
        self.mtt_algorithm = params['mtt_algorithm']
        self.PG = float(params['PG'])
        self.PD = float(params['PD'])
        self.gv = float(params['gv'])
        self.particle_size = float(params['particle_len'])
        self.min_len = float(params['min_trk_len'])

        vid_format = params['video_input'].split(sep='.')[-1]
        if vid_format == 'tif':
            tiff = tifffile.TiffFile(params['video_input'])
            tiff_resolution = tiff.pages[0].tags['XResolution'].value
            if self.um_per_px is None:
                self.um_per_px = tiff_resolution[1] / tiff_resolution[0]
            self.sequence = tiff.asarray()
        else:
            sequence_list = mimread(params['video_input'])
            self.sequence = np.array(sequence_list)
        self.algorithms = ['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF']
        self.out_dir = params['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)

        config = self.__dict__.copy()
        del config['octave']
        del config['algorithms']
        del config['sequence']
        with open(os.path.join(self.out_dir, 'config.txt'), 'w') as file:
            print(config, file=file)

    def detect(self):
        """
        Detects all particles in the video sequence and saves the results to a .csv.
        Returns:
            detections (pd.DataFrame): Dataframe with the detection results.
        """
        detections_file = os.path.join(self.out_dir, 'detections.csv')
        if self.detection_algorithm == 'Fluorescence':
            # Python implementation for segmentation and detection
            detections = evaluation(self.sequence, self.um_per_px)
            detections.to_csv(detections_file)
        elif self.detection_algorithm == 'Brightfield':
            # Python implementation for segmentation and detection (campo claro)
            detections = gray_evaluation(self.sequence)
            detections.to_csv(detections_file)
        elif self.detection_algorithm == 'Octave':
            # Urbano matlab implementation for segmentation and detection
            num_frames = self.sequence.shape[0]
            mimwrite('tmp.mp4', self.sequence, format='mp4', fps=self.fps)
            self.octave.Detector(detections_file, 'tmp.mp4', num_frames)
            os.remove('tmp.mp4')
            detections = pd.read_csv(detections_file)
        return detections

    def track(self):
        """
        Detects all trajectories in the video sequence and saves the results to a .csv.
        Returns:
            tracks (pd.DataFrame): Dataframe with the tracking results.
        """
        detections_file = os.path.join(self.out_dir, 'detections.csv')
        tracks_file = os.path.join(self.out_dir, 'trajectories.csv')

        mp4_video = 'tmp.mp4'
        output_video = 'output.mp4'
        mimwrite(mp4_video, self.sequence, format='mp4', fps=self.fps)

        # opcionales del código de matlab
        save_movie = 0
        plot_results = 0
        snap_shot = 0
        plot_track_results = 0
        analyze_motility = 0

        reformat_detections_file = self.detection_algorithm != 'Octave'
        num_frames = self.sequence.shape[0]
        ROIx = self.sequence.shape[2]
        ROIy = self.sequence.shape[1]

        mtt_algorithm = self.algorithms.index(self.mtt_algorithm) + 1
        particle_size = self.particle_size
        self.octave.Tracker(detections_file, mp4_video, output_video, tracks_file, int(reformat_detections_file),
                            num_frames, self.fps, self.um_per_px, ROIx, ROIy, mtt_algorithm,  self.PG, self.PD, self.gv,
                            particle_size, plot_results, save_movie, snap_shot, plot_track_results, analyze_motility,
                            nout=0)
        self.octave.clear_all(nout=0)

        tracks = pd.read_csv(tracks_file)
        tracks.columns = ['id', 'x', 'y', 'frame']
        tracks['fluorescence'] = np.nan
        tracks = tracks[['id', 'x', 'y', 'fluorescence', 'frame']]
        tracks[['x', 'y']] = tracks[['x', 'y']] / self.um_per_px

        # fluorescence
        if self.detection_algorithm != 'Brightfield':
            detections = pd.read_csv(detections_file)
            tracks = add_fluorescence_to_tracks(detections, tracks)

        # filter short tracks
        track_ids = tracks['id'].unique()
        if self.min_len:
            for track_id in track_ids:
                track = tracks[tracks['id'] == track_id]
                if len(track) < self.min_len:
                    tracks[tracks['id'] == track_id] = -1
        tracks = tracks[tracks['id'] != -1]
        os.remove(mp4_video)
        tracks.to_csv(tracks_file, index=False)
        return tracks

    def save_vid(self):
        tracks_file = os.path.join(self.out_dir, 'trajectories.csv')
        video_file = os.path.join(self.out_dir, 'trajectories.mp4')

        tracks = pd.read_csv(tracks_file)
        tracks_array = tracks.to_numpy()
        tracks_array[np.isnan(tracks_array)] = 0
        tracks_array = tracks_array[tracks_array[:, 4] < tracks_array[:, 4].max()]
        dets = pd.read_csv(os.path.join(self.out_dir, 'detections.csv'))
        sequence_tracks = draw_tracks(self.sequence, tracks_array, text=(self.detection_algorithm != 2))
        vid_sequence = draw_dets(sequence_tracks, dets)

        mimwrite(video_file, vid_sequence, format='mp4', fps=self.fps)

    def who_measures(self):
        tracks_file = os.path.join(self.out_dir, 'trajectories.csv')
        who_file = os.path.join(self.out_dir, 'who_measures.csv')
        get_casa_measures(tracks_file, who_file, self.um_per_px, self.fps)

    def who_classification(self):
        who_file = os.path.join(self.out_dir, 'who_measures.csv')
        classification_file = os.path.join(self.out_dir, 'who_classification.csv')
        df_measures = pd.read_csv(who_file)
        df_classified = classification(df_measures)
        df_classified.to_csv(classification_file)
