from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from pprint import PrettyPrinter
# pd.options.mode.chained_assignment = 'raise'


def fix_particles_oustide(X, M, N):
    # Se guarda en esta lista las partículas que están dentro del rango posible
    x_grater_0 = X.x > 0
    x_smaller_M = X.x < M
    y_grater_0 = X.y > 0
    y_smaller_N = X.y < N
    X_fix = X[np.logical_and(np.logical_and(x_grater_0, x_smaller_M), np.logical_and(y_grater_0, y_smaller_N))]

    return X_fix


def distance_between_two_tracks(track_a, track_b, max_dist):
    """
    Toma como entrada dos trayectorias y calcula la distancia entre ellas.
    d(track_a, track_b) = sum_{t=0}^{t=T-1} ||track_a(t)-track_b(t)||_{2}
    Parameters:
        track_a: dataframe que contiene las columnas (x, y, frame).
        track_b: igual que track_x
        max_dist: distancia máxima, si la distancia entre dos puntos es mayor a la distancia máxima se considera un
                    error en la asignacion.
    Returns:
        distance (float): distancia entre las dos trayectorias.
    """

    # check for empty tracks:
    if track_a['frame'].isnull().all() and not track_b['frame'].isnull().all():
        distance = len(track_b['frame']) * max_dist
        return distance
    elif track_b['frame'].isnull().all() and not track_a['frame'].isnull().all():
        distance = len(track_a['frame']) * max_dist
        return distance
    elif track_a['frame'].isnull().all() and track_b['frame'].isnull().all():
        distance = np.nan
        return distance

    # ------------------------------------------------------------------------------------------------------------------
    # old version:
    #    min_frame_a = track_a['frame'].min()
    #    min_frame_b = track_b['frame'].min()
    #    min_frame = int(min(min_frame_a, min_frame_b))
    #    max_frame_a = track_a['frame'].max()
    #    max_frame_b = track_b['frame'].max()
    #    max_frame = int(max(max_frame_a, max_frame_b))
    #
    #    distance = 0
    #    for frame in range(min_frame, max_frame+1):
    #        if frame < min_frame_a:
    #            distance += max_dist
    #        elif frame < min_frame_b:
    #            distance += max_dist
    #        elif frame > max_frame_a:
    #            distance += max_dist
    #        elif frame > max_frame_b:
    #            distance += max_dist
    #        else:
    #            coord_track_a = track_a[track_a['frame'] == frame][['x', 'y']].to_numpy().squeeze()
    #            coord_track_b = track_b[track_b['frame'] == frame][['x', 'y']].to_numpy().squeeze()
    #            l2_distance = np.sqrt((coord_track_a[0] - coord_track_b[0]) ** 2 +
    #                                  (coord_track_a[1] - coord_track_b[1]) ** 2)
    #            distance += min(l2_distance, max_dist)
    # ------------------------------------------------------------------------------------------------------------------

    first_frame_a = track_a['frame'].min()
    first_frame_b = track_b['frame'].min()
    first_frame = int(max(first_frame_a, first_frame_b))

    last_frame_a = track_a['frame'].max()
    last_frame_b = track_b['frame'].max()
    last_frame = int(min(last_frame_a, last_frame_b))


    distance = 0
    distance += (first_frame - first_frame_a)*max_dist
    distance += (first_frame - first_frame_b)*max_dist
    distance += (last_frame_a - last_frame)*max_dist
    distance += (last_frame_b - last_frame)*max_dist

    #print('BEFORE')
    #print('track_b', track_b['frame'].unique())
    #print('track_a', track_a['frame'].unique())
    #print('AFTER')
    track_a = track_a[track_a['frame'] >= first_frame]
    track_a = track_a[track_a['frame'] <= last_frame]
    track_b = track_b[track_b['frame'] >= first_frame]
    track_b = track_b[track_b['frame'] <= last_frame]

    union_index = np.isin(track_a['frame'], track_b['frame'].unique())
    track_a = track_a[union_index]
    #print('track_b', track_b['frame'].unique())
    #print('track_a', track_a['frame'].unique())
    distance += np.sum(~union_index)*max_dist

    union_index = np.isin(track_b['frame'], track_a['frame'].unique())
    track_b = track_b[union_index]
    # print('track_b', track_b['frame'].unique())
    # print('track_a', track_a['frame'].unique())
    distance += np.sum(~union_index) * max_dist

    coords_a = track_a.loc[:, ['x', 'y']].to_numpy()
    coords_b = track_b.loc[:, ['x', 'y']].to_numpy()

    distances = np.linalg.norm((coords_a - coords_b), axis=1)
    distance += np.sum(np.minimum(distances, max_dist))
    # print(distances)
    return distance


def get_optimal_assignment(X, Y, max_dist):
    """
    Determina el subconjunto Y_opt de Y que cumple dist(X,Y_opt) = min(X, Y*), siendo Y* cualquier subconjunto de Y.
    Parameters:
        X (array): De dimensiones (N, 2), siendo N la cantidad de coordenadas.
        Y (array): De dimensiones (M+N, 2), siendo N la cantidad de coordenadas.
        max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.
                        max_dist debería ser del órden del doble del tamaño promedio de las partículas.

    Returns:
        Y_opt: Subconjunto de Y que minimiza la distancia entre X e Y_opt.
    """
    n_X = X.shape[0]
    n_Y = Y.shape[0]

    cost = np.zeros([n_X, n_Y])
    for i in range(n_X):
        for j in range(n_Y):
            if Y[j, 0] == -1:
                cost[i, j] = max_dist
            else:
                cost[i, j] = np.linalg.norm(X[i, :2] - Y[j, :2], 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    Y_opt = np.zeros(X.shape)
    for i in range(n_X):
        Y_opt[i, :] = Y[col_ind[i], :]

    return Y_opt


def get_optimal_track_assignment(tracks_a, tracks_b, max_dist):
    """
    Determina el subconjunto Y_opt de Y que cumple dist(X,Y_opt) = min(X, Y*), siendo Y* cualquier subconjunto de Y.
    Parameters:
        tracks_a (DataFrame): Con las columnas ['id', 'x', 'y', 'frame'].
        tracks_b (DataFrame): Columnas ['id', 'x', 'y', 'frame'].
        max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.
                        max_dist debería ser del órden del doble del tamaño promedio de las partículas.
    Returns:
        tracks_a (DataFrame): Se agrega la columna 'opt_track_id' al dataframe de entrada tracks_a, indicando el
                                id_track de track_b asignado.
        tracks_b (DataFrame): Se agrega la columna 'opt_track_id' al dataframe de entrada tracks_b, indicando el
                                id_track de track_a asignado.
        cost (list): Distancias de las trayectorias asignadas. Ordenado ...
    """
    ids_a = tracks_a['id'].unique()
    ids_b = tracks_b['id'].unique()
    num_tracks_a = len(ids_a)
    num_tracks_b = len(ids_b)

    tracks_a_grouped = tracks_a.groupby('id')
    tracks_b_grouped = tracks_b.groupby('id')

    cost = np.zeros([num_tracks_a, num_tracks_b])
    for i in range(num_tracks_a):
        for j in range(num_tracks_b):
            cost[i, j] = distance_between_two_tracks(tracks_a_grouped.get_group(ids_a[i]),
                                                     tracks_b_grouped.get_group(ids_b[j]),
                                                     max_dist)

    row_ind, col_ind = linear_sum_assignment(cost)
    # dist = cost[row_ind, col_ind].sum()

    for i in range(len(col_ind)):
        tracks_b.loc[tracks_b['id'] == ids_b[col_ind[i]], 'opt_track_id'] = ids_a[row_ind[i]]
        tracks_a.loc[tracks_a['id'] == ids_a[row_ind[i]], 'opt_track_id'] = ids_b[col_ind[i]]

    return tracks_a, tracks_b, cost[row_ind, col_ind]


def error_measures(ground_truth_df, detected_df, max_dist):
    """
    Parameters:
        ground_truth_df (df(id, x, y, total_pixels, mask, frame)): Coordenadas ground truth de las partículas.
        detected_df (df(id, x, y, total_pixels, mask, frame)): Coordenadas medidas de las partículas.
        max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.
                        max_dist debería ser del órden del doble del tamaño promedio de las partículas.

    Returns:
        TP (int): Verdaderos Positivos
        FN (int): Falsos Negativos
        FP (int): Falsos Positivos
        JSC (int): Índice de Jaccard
    """

    TP = 0
    FN = 0
    FP = 0

    total_frames = int(ground_truth_df['frame'].max())

    for f in range(total_frames + 1):
        ground_truth_f_df = ground_truth_df[ground_truth_df.frame.astype(int) == f]
        detected_f_df = detected_df[detected_df.frame.astype(int) == f]

        ground_truth_np = ground_truth_f_df[['x', 'y', 'frame']].to_numpy()
        detected_np = detected_f_df[['x', 'y', 'frame']].to_numpy()

        detected_extended_np = np.concatenate((detected_np, (-1) * np.ones(ground_truth_np.shape)), axis=0)
        detected_opt_np = get_optimal_assignment(ground_truth_np, detected_extended_np, max_dist)

        TP += len(detected_opt_np[detected_opt_np[:, 0] != -1])
        FN += len(detected_opt_np[detected_opt_np[:, 0] == -1])
        FP += len(detected_np[:, 0]) - len(detected_opt_np[detected_opt_np[:, 0] != -1])

    JSC = TP / (TP + FN + FP)

    return TP, FN, FP, JSC


def track_set_error(ground_truth, estimated_tracks, max_dist):
    """
    Toma como entrada del conjunto de trayectorias a evaluar y el ground truth con que comparar.
    Parameters:
        ground_truth: dataframe que contiene las columnas (id, x, y, frame).
        estimated_tracks: dataframe que contiene las columnas (id, x, y, frame).
        max_dist: distancia máxima, si la distancia entre dos puntos es mayor a la distancia máxima se considera un
                    error en la asignacion.
    Returns:
        performance_measures (dict): con las siguientes keys:
                alpha (float): Definido en "Performance measures". Entre 0 y 1, es 1 si los conjuntos son iguales, y 0
                                en el mayor error posible.
                                alpha(ground_truth, tracks) = 1 - d(ground_truth, tracks)/d(ground_truth, dummy_tracks)
                beta (float): Definido en "Performance measures". Entre 0 y alpha, es alpha si no hay tracks erroneas
                                y converge a cero a medida que el número aumenta.
                                beta(ground_truth, tracks) = (d(ground_truth, dummy_tracks) - d(ground_truth, tracks)) /
                                                        (d(ground_truth, dummy_tracks) + d(right_tracks, dummy_tracks))
                TP Tracks (int): True Positives. Número de trayectorias correctas de tracks.
                FN Tracks (int): False Negatives. Número de trayectorias de ground truth que no se encuentran en tracks.
                FP Tracks (int): False Positives. Número de trayectorias de tracks que no corresponden a ninguna de
                                            ground_truth.
                JSC Tracks (float): Índice de Jaccard. JSC = TP/(TP + FN + FP)
    """

    dummy_tracks = {'id': -ground_truth['id'].unique()}
    dummy_tracks = pd.DataFrame(data=dummy_tracks, columns=['id', 'frame'])

    tracks_extended = pd.concat([estimated_tracks, dummy_tracks])
    # Se calcula la distancia del conjunto de tracks al ground truth
    ground_truth, tracks_extended, opt_distances = get_optimal_track_assignment(ground_truth, tracks_extended, max_dist)
    opt_distance = opt_distances.sum()

    # La máxima distancia posible entre tracks y ground_truth, es la distancia entre ground truth
    # y un conjunto de tracks vacías.
    _, _, max_distances = get_optimal_track_assignment(ground_truth, dummy_tracks, max_dist)
    max_distance = max_distances.sum()

    # tracks not assigned to ground_truth tracks
    wrong_tracks = tracks_extended[tracks_extended['opt_track_id'].isnull()]
    wrong_tracks = wrong_tracks[wrong_tracks['id'] > 0]                                # only non dummy tracks
    dummy_tracks = {'id': -wrong_tracks['id'].unique()}
    dummy_tracks = pd.DataFrame(data=dummy_tracks, columns=['id', 'frame'])
    _, _, wrong_max_distances = get_optimal_track_assignment(wrong_tracks, dummy_tracks, max_dist)
    wrong_max_distance = wrong_max_distances.sum()

    # tracks assigned to ground_truth tracks
    assigned_tracks = tracks_extended[~tracks_extended['opt_track_id'].isnull()]
    right_tracks = assigned_tracks[assigned_tracks['id'] > 0]                           # only non dummy assigned tracks
    _, _, right_distances = get_optimal_track_assignment(right_tracks, ground_truth, max_dist)
    # print('distance: ', opt_distance)
    # print('max_dist: ', max_distance)

    alpha = 1 - opt_distance/max_distance
    beta = (max_distance - opt_distance)/(max_distance + wrong_max_distance)
    # number non dummy tracks assigned to ground_truth tracks:
    TP = len(right_tracks['id'].unique())
    # number of dummy tracks assigned to ground_truth tracks:
    FN = len(assigned_tracks[assigned_tracks['id'] < 0]['id'].unique())
    # number of non dummy tracks not assigned to ground_truth tracks"
    FP = len(wrong_tracks['id'].unique())
    JSC = TP/(TP + FN + FP)

    rmse = np.sqrt(np.mean(right_distances ** 2))
    min = np.min(right_distances)
    max = np.max(right_distances)
    sd = np.std(right_distances)

    # Number of right positions in tracks assigned to ground truth tracks.
    TP_positions = 0
    # TP = assigned_tracks[assigned_tracks['id'] > 0].shape[0]

    estimated_track_grouped = tracks_extended.groupby('id')
    gt_grouped = ground_truth.groupby('id')
    for opt_id in ground_truth['opt_track_id'].unique():
        if opt_id > 0:
            estimated_track = estimated_track_grouped.get_group(opt_id)
            gt_id = estimated_track['opt_track_id'].unique()[0]
            gt_track = gt_grouped.get_group(gt_id)

            track_frames = estimated_track['frame'].unique()
            gt_track_frames = gt_track['frame'].unique()

            first_frame = np.max(np.array([track_frames.min(), gt_track_frames.min()]))
            last_frame = np.min(np.array([track_frames.max(), gt_track_frames.max()]))

            estimated_track = estimated_track[estimated_track['frame'] >= first_frame]
            estimated_track = estimated_track[estimated_track['frame'] <= last_frame]
            gt_track = gt_track[gt_track['frame'] >= first_frame]
            gt_track = gt_track[gt_track['frame'] <= last_frame]

            union_index = np.isin(gt_track['frame'], estimated_track['frame'].unique())
            gt_track = gt_track[union_index]
            union_index = np.isin(estimated_track['frame'], gt_track['frame'].unique())
            estimated_track = estimated_track[union_index]

            est_coords = estimated_track.loc[:, ['x', 'y']].to_numpy()
            gt_coords = gt_track.loc[:, ['x', 'y']].to_numpy()

            dists = np.linalg.norm((est_coords - gt_coords), axis=1)
            TP_positions += np.sum(dists < max_dist)

    # Number of positions assigned to dummy tracks:
    FN_positions = ground_truth[ground_truth['opt_track_id'] < 0].shape[0]
    # Number of positions of tracks not assigned to ground truth tracks:
    FP_positions = wrong_tracks.shape[0]

    JSC_positions = TP_positions/(TP_positions + FN_positions + FP_positions)

    performance_measures = {
        'alpha': alpha,
        'beta': beta,
        'TP Tracks': TP,
        'FN Tracks': FN,
        'FP Tracks': FP,
        'JSC Tracks': JSC,
        'RMSE': rmse,
        'Min': min,
        'Max': max,
        'SD': sd,
        'TP Positions': TP_positions,
        'FN Positions': FN_positions,
        'FP Positions': FP_positions,
        'JSC Positions': JSC_positions
    }
    return performance_measures
# ----------------------------------------------------------------------------------------------------------------------
# PRUEBA:

# N = 15
# F = 5

# X = np.zeros((N*F,3))
# Y = np.zeros((N*F,3))


# for n in range(N):
# 	for f in range(F):
# 		X[n*F+f,:] = [250+np.random.normal(0, 20), 250+np.random.normal(0, 20), f]
# 		Y[n*F+f,:] = [500+np.random.normal(0, 20), 500+np.random.normal(0, 20), f]

# for i in range(30):
# 	Y[i,:] = X[i,:]
# df_X = pd.DataFrame(X,columns = ['x','y','frame'])
# df_Y = pd.DataFrame(Y,columns = ['x','y','frame'])

# print(df_X)
# print(df_Y)

# TP, FN, FP, JSC = error_measures(df_X, df_Y, 512, 512, 30)
# print(TP, FN, FP, JSC)

# ----------------------------------------------------------------------------------------------------------------------
# PRUEBA: distance_between_two_tracks()
#
# tracks = pd.read_csv('tracks.csv')
# print(tracks.head())
# track_a = tracks[tracks['id'] == 2]
# track_b = tracks[tracks['id'] == 3]
# print(track_a.head())
# print(track_b.head())
# dist = distance_between_two_tracks(track_a, track_b, 0)
# print(dist)
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# PRUEBA: track_set_error()
#
# tracks_csv = pd.read_csv('tracks_enn_jpdaf.csv')
# gt_tracks = pd.read_csv('_data.csv')
# gt_tracks = gt_tracks[gt_tracks['frame'] < 50]
# tracks_csv = tracks_csv[tracks_csv['frame'] < 50]
# gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
# print('---------------------------------------------------ENN JPDAF---------------------------------------------------
# ')
# print('tracks: \n', 'shape:', tracks_csv.shape, '\n', tracks_csv.head())
# print('----------------------------------------------------')
# print('gt: \n', 'shape:', gt_tracks.shape, '\n', gt_tracks.head())
# print('----------------------------------------------------')

# error = track_set_error(gt_tracks, tracks_csv, 40)
# print('\n Performance Measures:')
# PrettyPrinter(sort_dicts=False).pprint(error)

# tracks_csv = pd.read_csv('tracks_NN.csv')
# tracks_csv = tracks_csv[tracks_csv['frame'] < 50]
# print('-------------------------------------------------------NN------------------------------------------------------')
# print('tracks: \n', 'shape:', tracks_csv.shape, '\n', tracks_csv.head())
# print('----------------------------------------------------')
# print('gt: \n', 'shape:', gt_tracks.shape, '\n', gt_tracks.head())
# print('----------------------------------------------------')

# error = track_set_error(gt_tracks, tracks_csv, 40)
# print('\n Performance Measures:')
# PrettyPrinter(sort_dicts=False).pprint(error)

# tracks_a = {
#    'id': [3, 4, 4, 4, 55, 55, 3],
#    'x': [3, 45, 47, 50, 2, 5, 44],
#    'y': [3, 45, 42, 39, 10, 15, 55],
#    'frame': [4, 1, 2, 3, 1, 2, 5]
# }
# tracks_a = pd.DataFrame(data=tracks_a)
# print('tracks_a: \n', tracks_a)

# tracks_b = {
#     'id': [1, 1, 2, 2, 1],
#     'x': [45, 47, 2, 5, 50],
#     'y': [45, 42, 10, 15, 39],
#     'frame': [1, 2, 1, 2, 3]
# }
# tracks_b = pd.DataFrame(data=tracks_b)
# print('tracks_b: \n', tracks_b)

# err = track_set_error(tracks_a, tracks_b, 10)
# print('error: ', err)
# ----------------------------------------------------------------------------------------------------------------------
