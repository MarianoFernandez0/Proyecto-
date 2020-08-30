from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":
    indir = 'dataset_1(40Hz)_data.csv'
    out_dir = 'out_dataset'
    os.makedirs(out_dir, exist_ok=True)
    res = 1
    fps = 40
    get_casa_measures(indir, out_dir, res, fps)
