import os
import sys
import shutil
import pandas as pd
import numpy as np
import tifffile
from imageio import mimwrite, mimread
from tool.src.detection.evaluation import evaluation
from tool.src.detection.gray_detection import gray_evaluation
from tool.src.fluorescence.add_fluorescence import add_fluorescence_to_tracks
from tool.src.who_measures.get_who_measures import get_casa_measures
from tool.src.classification.classification_WHO import classification

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
    TOOL_PATH = application_path.split(sep='/')
    TOOL_PATH.pop(-1)
    TOOL_PATH.pop(-1)
    TOOL_PATH.insert(0, '/')
    os.environ["OCTAVE_KERNEL_JSON"] = os.path.join('/', *TOOL_PATH, 'octave_kernel/kernel.json')
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
    TOOL_PATH = application_path.split(sep='/')
    TOOL_PATH.pop(-1)
    TOOL_PATH.pop(-1)
import oct2py
TOOL_PATH[0] = '/'
os.makedirs(os.path.join(*TOOL_PATH, 'tmp'), exist_ok=True)
octave = oct2py.Oct2Py(temp_dir=os.path.join(*TOOL_PATH, 'tmp'))
octave.addpath(os.path.join(*TOOL_PATH, 'src/SpermTrackingProject'))
octave.addpath(os.path.join(*TOOL_PATH, 'src/oct2py'))


class Tracker:
    """
    Attributes:
        video_file (str): Input video sequence.
        fps (int): Frame frequency.
        px2um (float): Scale of the image.

        detection_algorithm (int): Detection algorithm.
                                    0 = MatLab implementation
                                    1 = Python implementation
        reformat_detections_file (int): Depends on the detection algorithm implementation.
                                            0 = MatLab implementation
                                            1 = Python implementaadd_fluorescence_to_trackstion
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

    """
    def __init__(self, params, octave_interpreter=octave):
        self.octave = octave_interpreter
        self.fps = int(params['fps'])

        if isinstance(params['px2um'], (int, float)) or params['px2um'].replace('.', '', 1).isdigit():
            self.px2um = float(params['px2um'])
        else:
            self.px2um = None

        self.detection_algorithm = int(params['detection_algorithm'])
        self.mtt_algorithm = int(params['mtt_algorithm'])
        self.PG = float(params['PG'])
        self.PD = float(params['PD'])
        self.gv = float(params['gv'])

        vid_format = params['video_input'].split(sep='.')[-1]
        if vid_format == 'tif':
            tiff = tifffile.TiffFile(params['video_input'])
            print(tiff.asarray().shape)
            print(params['video_input'])
            tiff_resolution = tiff.pages[0].tags['XResolution'].value
            if self.px2um is None:
                self.px2um = tiff_resolution[1] / tiff_resolution[0]
            self.sequence = tiff.asarray()
            raise ValueError('TESTING')
        else:
            sequence_list = mimread(params['video_input'])
            self.sequence = np.array(sequence_list)

        self.outdir = params['outdir']
        self.basename = (params['video_input'].rsplit('.', 1)[-2]).split('/')[-1]

    def detect(self):
        """
        Detects all particles in the video sequence and saves the results to a .csv.
        Args:
            detections_file (str): Output csv with estimated detections.
        Returns:
            detections (pd.DataFrame): Dataframe with the detection results.

        """
        os.makedirs(self.outdir + '/detections', exist_ok=True)
        detections_file = self.outdir + "/detections/" + self.basename + '_detections.csv'
        detections = None
        if self.detection_algorithm == 1:
            # Python implementation for segmentation and detection
            detections = evaluation(self.sequence, self.px2um)
            detections.to_csv(detections_file)
        elif self.detection_algorithm == 2:
            # Python implementation for segmentation and detection (campo claro)
            detections = gray_evaluation(self.sequence)
            detections.to_csv(detections_file)
        elif self.detection_algorithm == 0:
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
        Args:
            detections_file (str): Output csv with estimated detections.
            tracks_file (str): Output csv with estimated tracks.
        Returns:
            tracks (pd.DataFrame): Dataframe with the tracking results.
        """
        os.makedirs(self.outdir + "/tracks", exist_ok=True)
        tracks_file = self.outdir + "/tracks/" + self.basename + '_tracks.csv'
        detections_file = self.outdir + "/detections/" + self.basename + '_detections.csv'

        mp4_video = 'tmp.mp4'
        output_video = self.basename + '.mp4'
        mimwrite(mp4_video, self.sequence, format='mp4', fps=self.fps)

        # opcionales del código de matlab
        save_movie = 0
        plot_results = 0
        snap_shot = 0
        plot_track_results = 0
        analyze_motility = 0

        reformat_detections_file = self.detection_algorithm
        num_frames = self.sequence.shape[0]
        ROIx = self.sequence.shape[2]
        ROIy = self.sequence.shape[1]

        self.octave.Tracker(detections_file, mp4_video, output_video, tracks_file, reformat_detections_file, num_frames,
                            self.fps, self.px2um, ROIx, ROIy, self.mtt_algorithm,  self.PG, self.PD, self.gv,
                            plot_results, save_movie, snap_shot, plot_track_results, analyze_motility, nout=0)
        self.octave.clear_all(nout=0)

        tracks = pd.read_csv(tracks_file)
        tracks.columns = ['id', 'x', 'y', 'frame']
        tracks['fluorescence'] = np.nan
        tracks = tracks[['id', 'x', 'y', 'fluorescence', 'frame']]
        tracks[['x', 'y']] = tracks[['x', 'y']] / self.px2um

        # fluorescence
        if self.detection_algorithm != 2:
            detections = pd.read_csv(detections_file)
            tracks = add_fluorescence_to_tracks(detections, tracks)
        tracks.to_csv(tracks_file, index=False)
        os.remove(mp4_video)

        return tracks

    def get_who_measures(self):
        tracks_file = os.path.join(self.outdir + "/tracks/", self.basename + '_tracks.csv')
        os.makedirs(self.outdir + '/who_measures', exist_ok=True)
        who_file = self.outdir + '/who_measures'
        get_casa_measures(tracks_file, who_file, self.px2um, self.fps)

    def who_classification(self):
        files = os.listdir(self.outdir + '/who_measures')
        file = self.outdir + '/who_measures/' + str([f for f in files if f.endswith('.csv')][0])
        df_measures = pd.read_csv(file)
        df_classified = classification(df_measures)
        os.makedirs(self.outdir + '/who_classification', exist_ok=True)
        classification_file = self.outdir + '/who_classification/' + self.basename + '_who_classification.csv'
        df_classified.to_csv(classification_file)


def delete_tmp():
    shutil.rmtree(os.path.join(*TOOL_PATH, 'tmp'))
