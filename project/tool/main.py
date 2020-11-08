import json
import argparse
from src.tracking.tracking import Tracker
from src.lif_files_processing import run_for_lif

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate tracks from video sequence.')
    parser.add_argument('--config', default='input/configs/config.json', type=str,
                        help='Config file with the tracking parameters.')
    parser.add_argument('--save_vid', action='store_true', default=True, help='Save video with drawn tracks.')
    parser.add_argument('--lif', action='store_true',  default=False)
    parser.add_argument('--lif_dir', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.lif:
        run_for_lif(args.lif_dir, config)
    else:
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
