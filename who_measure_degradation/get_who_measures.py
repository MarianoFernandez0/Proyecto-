import pandas as pd
from get_carac import get_carac
import os
import numpy as np

def read_csv_file(df):
    TRACK_ID = []
    X = []
    Y = []
    F = []
    fluo = []
    for id in np.unique(df['id'].to_numpy()):
        TRACK_ID.append(id)
        new_out = df.loc[df['id'] == id]
        X.append(new_out['x'].to_list())
        Y.append(new_out['y'].to_list())
        F.append((new_out['frame']).to_list())
        fluo.append(new_out['fluorescence'].to_numpy())

    return X, Y, F, TRACK_ID, fluo

def get_casa_measures(in_dir, out_dir, scale, fps):
    # leer archivo csv con los tracks
    out = pd.read_csv(in_dir)
    # reordenar dataframe
    out_rearranged = out.sort_values(by=['id', 'frame'])

    # cambiar la escala
    out_rearranged["x"] = out_rearranged["x"] * scale
    out_rearranged["y"] = out_rearranged["y"] * scale

    # usar funcion read_csv para obtener las listas X, Y, T, TRACK_ID
    X, Y, F, TRACK_ID, fluo = read_csv_file(out_rearranged)

    # calcular caracteristicas CASA
    CARAC_WHO = get_carac(TRACK_ID, X, Y, F, fps, fluo, min_detections=3)

    # guardar parametros
    param_who = pd.DataFrame(CARAC_WHO)
    param_who.columns = ['track_id', 'vcl', 'vsl', 'vap_mean', 'vap_std', 'alh_mean', 'alh_std', 'lin', 'wob', 'str',
                         'bcf_mean', 'bcf_std', 'mad', 'fluo']
    os.makedirs(out_dir, exist_ok=True)
    param_who.to_csv(out_dir + '/' + (in_dir.split('/')[-1]).split('.')[0] + '_WHO.csv', index=False)


if __name__ == "__main__":
    indir = 'dataset_1_7Hz_ennjpdaf.csv'
    outdir = 'out_test'
    get_casa_measures(indir, outdir, 1.5)
