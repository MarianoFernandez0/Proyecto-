import argparse
import json
from tool.src.tracking.tracking import Tracker, delete_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='./root/project/data/configs/tracking_config_test.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', default=False, help='Save video with drawn tracks.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    tracker = Tracker(params=config)
    tracker.detect(detections_file=config['detections_csv'])
    tracker.track(detections_file=config['detections_csv'], tracks_file=config['tracks_csv'])
    tracker.save_vid(tracks_file=config['tracks_csv'], video_file=config['tracks_video'])
    delete_tmp()
