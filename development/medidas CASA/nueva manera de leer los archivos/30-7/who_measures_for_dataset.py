from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":

    indir = 'results_with_fluorescence'
    out_base = 'who_measures'
    datasets = [directory for directory in os.listdir(indir) if os.path.isdir(os.path.join(indir, directory))]
    datasets = ['dataset_1']
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