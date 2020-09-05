from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":
    indir = 'tracks1026.csv'
    out_dir = 'out_dataset_1026'
    os.makedirs(out_dir, exist_ok=True)
    res = 2.6387
    fps = 30
    get_casa_measures(indir, out_dir, res, fps)
