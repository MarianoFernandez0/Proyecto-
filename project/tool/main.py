import json
import argparse
from src.tracking.tracking import Tracker


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='input/configs/config.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', default=True, help='Save video with drawn tracks.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    tracker = Tracker(params=config)
    # detect
    tracker.detect()
    # track
    tracks = tracker.track()
    # save_vid
    tracker.save_vid()
    # who_measures
    tracker.who_measures()
    # who_classification
    tracker.who_classification()
