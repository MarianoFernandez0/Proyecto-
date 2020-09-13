from get_who_measures import get_casa_measures
import os

if __name__ == "__main__":
    indir = 'tracks.csv'
    out_dir = 'out_dataset'
    os.makedirs(out_dir, exist_ok=True)
    res = 0.813
    fps = 30
    get_casa_measures(indir, out_dir, res, fps)
