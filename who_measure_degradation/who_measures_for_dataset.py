from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":
    indir = 'dataset_29_8_2020'
    out_base = 'measures'
    res = 1
    fps = 0
    datasets = [directory for directory in os.listdir(indir) if os.path.isdir(os.path.join(indir, directory))]
    for dataset in datasets:
        out_dir = os.path.join(out_base, dataset)
        tracking_path = os.path.join(indir, dataset, 'tracking_results')
        result_files = [file for file in os.listdir(tracking_path) if file.endswith('.csv') and 'ennjpdaf.csv' in file.split('_')]
        result_files.sort()
        for result in result_files:
            print(dataset + " " + result)
            fps = float((result.split("Hz")[0]).split("_")[-1])
            track_path = os.path.join(tracking_path, result)
            get_casa_measures(track_path, out_dir, res, fps)
        print()
