from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":
    indir = 'dataset_4(60Hz)_data.csv'
    out_dir = 'out_dataset'
    os.makedirs(out_dir, exist_ok=True)
    res = 1
    fps = 60
    get_casa_measures(indir, out_dir, res, fps)
