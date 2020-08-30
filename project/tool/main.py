import argparse
import os
import json
from oct2py import octave
from tool.src.tracking.tracking import Tracker

# add .m files to octave path
current_path = os.path.realpath(__file__).split(sep='/')
current_path.pop(-1)
current_path = '/' + os.path.join(*current_path)
octave.addpath(current_path + '/src/SpermTrackingProject')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='input/configs/tracking_config_test.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', default=True, help='Save video with drawn tracks.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    tracker = Tracker(params=config, octave_interpreter=octave)
    tracker.detect(detections_file=config['detections_csv'])
    tracker.track(detections_file=config['detections_csv'], tracks_file=config['tracks_csv'])
    tracker.save_vid(tracks_file=config['tracks_csv'], video_file=config['tracks_video'])
