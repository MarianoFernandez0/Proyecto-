import pandas as pd
from read_csv_file import read_csv_file
from get_carac import get_carac
import os
import numpy as np


def get_casa_measures(in_dir, out_dir, scale, fps):
    # leer archivo csv con los tracks
    out = pd.read_csv(in_dir)
    # reordenar dataframe
    out_rearranged = out.sort_values(by=['id', 'frame'])

    # cambiar la escala
    out_rearranged["x"] = out_rearranged["x"] * scale
    out_rearranged["y"] = out_rearranged["y"] * scale

    # usar funcion read_csv para obtener las listas X, Y, T, TRACK_ID
    X, Y, F, TRACK_ID = read_csv_file(out_rearranged)

    # calcular caracteristicas CASA
    CARAC_WHO = get_carac(TRACK_ID, X, Y, F, fps, min_detections=3)

    # guardar parametros
    param_who = pd.DataFrame(CARAC_WHO)
    param_who.columns = ['track_id', 'vcl', 'vsl', 'vap_mean', 'vap_std', 'alh_mean', 'alh_std', 'lin', 'wob', 'stra',
                         'bcf_mean', 'bcf_std', 'mad']
    os.makedirs(out_dir, exist_ok=True)
    param_who.to_csv(out_dir + '/' + (in_dir.split('/')[-1]).split('.')[0] + '_WHO.csv', index=False)


if __name__ == "__main__":
    indir = 'dataset_1_7Hz_ennjpdaf.csv'
    outdir = 'out_test'
    get_casa_measures(indir, outdir, 1.5)
