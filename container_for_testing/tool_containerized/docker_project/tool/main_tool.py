import argparse
import json
from tool.src.tracking.tracking import Tracker, delete_tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='/data/input/config/config_tracking.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', help='Save video with drawn tracks.')
    args = parser.parse_args()



    with open(args.config, 'r') as f:
        config = json.load(f)

    tracker = Tracker(params=config)
    tracker.detect()
    tracker.track()
    tracker.get_who_measures()
    tracker.who_classification()
    delete_tmp()
