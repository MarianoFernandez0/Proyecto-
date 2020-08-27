from add_fluorescence import add_fluorescence_to_tracks
import os
from tqdm import tqdm
from argparse import ArgumentParser

# TODO: Paralelizar esto así no demora

if __name__ == "__main__":

    indir = 'datasets_octave_results'
    out_base = 'results_with_fluorescence'
    datasets = [directory for directory in os.listdir(indir) if os.path.isdir(os.path.join(indir, directory))]
    for dataset in datasets:
        out_dir = os.path.join(out_base, dataset)
        detections_path = os.path.join(indir, dataset, 'detection_results')
        tracking_path = os.path.join(indir, dataset, 'tracking_results')
        result_files = [file for file in os.listdir(tracking_path) if file.endswith('.csv')]
        for result in result_files:
            print(f'{dataset+" "+result}\r', end = "")
            track_path = os.path.join(tracking_path, result)
            detection_path = os.path.join(detections_path, result)
            add_fluorescence_to_tracks(detection_path, track_path, out_dir)
        print()