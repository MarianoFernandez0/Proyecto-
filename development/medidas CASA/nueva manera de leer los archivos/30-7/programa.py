import numpy as np
import pandas as pd
import math

from read_csv_file import read_csv_file
from get_carac import get_carac


# leer archivo csv con los tracks
out = pd.read_csv("out.csv")

# reordenar dataframe
out_rearranged = out.sort_values(by=['id', 'frame'])

# guardar una copia del dataframe ordenado
out_rearranged.to_csv('out-byid.csv',index=False)

# cambiar la escala
scale=1
out_rearranged["x"] = out_rearranged["x"] * scale
out_rearranged["y"] = out_rearranged["y"] * scale

# usar funcion read_csv para obtener las listas X, Y, T, TRACK_ID
X,Y,T,TRACK_ID = read_csv_file(out_rearranged)

# calcular caracteristicas CASA
CARAC_WHO = get_carac(X,Y,T,min_detections=3)

# guardar parametros
param_who = pd.DataFrame(CARAC_WHO)
param_who.columns = ['vcl', 'vsl', 'vap_mean', 'vap_std', 'alh_mean', 'alh_std', 'lin', 'wob', 'stra', 'bcf_mean', 'bcf_std', 'mad']
param_who.to_csv('param_who.csv',index=False)