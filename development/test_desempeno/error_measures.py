from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from pprint import PrettyPrinter
import time


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

    num_frames = max(track_a['frame'].to_numpy(dtype='int32').max(), track_b['frame'].to_numpy(dtype='int32').max()) + 1
    max_x = max(track_a['x'].to_numpy(dtype='int32').max(), track_b['x'].to_numpy(dtype='int32').max())
    max_y = max(track_a['y'].to_numpy(dtype='int32').max(), track_b['x'].to_numpy(dtype='int32').max())

    track_a_np = np.ones((2, num_frames))*max_x*max_y
    track_b_np = np.ones((2, num_frames))*max_x*max_y
    track_a_np[:, track_a['frame'].to_numpy(dtype='int32')] = track_a[['x', 'y']].to_numpy().T
    track_b_np[:, track_b['frame'].to_numpy(dtype='int32')] = track_b[['x', 'y']].to_numpy().T

    distances = np.linalg.norm((track_a_np - track_b_np), axis=0)
    distance = np.sum(np.minimum(distances, max_dist))

    return distance


def get_optimal_assignment(X, Y, max_dist):
    """
    Determina el subconjunto Y_opt de Y que cumple dist(X,Y_opt) = min(X, Y*), siendo Y* cualquier subconjunto de Y.
    Parameters:
        X (ndarray): De dimensiones (N, 2), siendo N la cantidad de coordenadas.
        Y (ndarray): De dimensiones (M+N, 2), siendo N la cantidad de coordenadas.
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


def pandas_tracks_to_numpy(tracks, num_frames, max_num):
    """
    Cambia el formato de un conjunto de tracks, de un DataFrame a un array de numpy.
    Parameters:
        tracks (DataFrame): Conjunto de tracks con las columnas ['x', 'y', 'frame', 'id']
        num_frames (int): Número de frames.
        max_num (int): Número a asignar en los puntos vacíos.
    Returns:
        tracks_new (ndarray): Numpy array con dimensiones (2, numframe, num_tracks).
        ids (ndarray): lookup table con los ids de las tracks.
    """
    ids = tracks['id'].unique()
    num_tracks = len(ids)

    tracks_grouped = tracks.groupby('id')
    tracks_new = np.ones((2, num_frames, num_tracks))*max_num

    for i in range(num_tracks):
        if ~(np.isnan(tracks_grouped.get_group(ids[i])[['x', 'y']].to_numpy())).any():
            tracks_new[:, tracks_grouped.get_group(ids[i])['frame'].to_numpy(dtype='int32'), i] = \
                tracks_grouped.get_group(ids[i])[['x', 'y']].to_numpy().T

    return tracks_new, ids


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
    num_frames = tracks_a['frame'].to_numpy(dtype='int32').max()+1
    max_x = max(tracks_a['x'].to_numpy(dtype='int32').max(), tracks_b['x'].to_numpy(dtype='int32').max())
    max_y = max(tracks_a['y'].to_numpy(dtype='int32').max(), tracks_b['x'].to_numpy(dtype='int32').max())

    tracks_a_np, ids_a = pandas_tracks_to_numpy(tracks_a, num_frames, max_x*max_y)
    tracks_b_np, ids_b = pandas_tracks_to_numpy(tracks_b, num_frames, max_x*max_y)
    num_tracks_a = len(ids_a)
    num_tracks_b = len(ids_b)

    cost = np.zeros([num_tracks_a, num_tracks_b])
    for i in range(num_tracks_a):
        for j in range(num_tracks_b):
            distances = np.linalg.norm((tracks_a_np[:, :, i] - tracks_b_np[:, :, j]), axis=0)
            cost[i, j] = np.sum(np.minimum(distances, max_dist))

    row_ind, col_ind = linear_sum_assignment(cost)

    tracks_a_new = tracks_a.copy()
    tracks_b_new = tracks_b.copy()

    for i in range(len(col_ind)):
        tracks_a_new.at[tracks_a['id'] == ids_a[row_ind[i]], 'opt_track_id'] = ids_b[col_ind[i]]
        tracks_b_new.at[tracks_b['id'] == ids_b[col_ind[i]], 'opt_track_id'] = ids_a[row_ind[i]]

    return tracks_a_new, tracks_b_new, cost[row_ind, col_ind]


def detection_error_measures(ground_truth_df, detected_df, max_dist):
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

    if any(estimated_tracks['id'].unique() == 0):
        estimated_tracks.id = estimated_tracks['id'] + 1  # define tracks id > 0

    dummy_tracks = {'id': -ground_truth['id'].unique()}
    dummy_tracks = pd.DataFrame(data=dummy_tracks, columns=['id', 'frame'])

    tracks_extended = pd.concat([estimated_tracks, dummy_tracks])

    # Se calcula la distancia entre conjunto de tracks estimadas y el de ground truth
    t0 = time.time()
    ground_truth, tracks_extended, opt_distances = get_optimal_track_assignment(ground_truth, tracks_extended, max_dist)
    t1 = time.time()
    print('Time to run optimal assignment: ', t1 - t0)
    opt_distance = opt_distances.sum()

    # La máxima distancia posible entre las tracks estimadas y el ground_truth
    max_distance = ground_truth.shape[0]*max_dist

    # tracks estimadas que no se asignaron a track de ground_truth
    wrong_tracks = tracks_extended[tracks_extended['opt_track_id'].isnull()]  # Tracks no asignadas
    wrong_tracks = wrong_tracks[wrong_tracks['id'] > 0]  # Tracks pertenecientes a estimated_tracks
    wrong_max_distance = wrong_tracks.shape[0]*max_dist

    # tracks estimadas asignadas a tracks de ground_truth
    assigned_tracks = tracks_extended[~tracks_extended['opt_track_id'].isnull()]   # Tracks asignadas
    right_tracks = assigned_tracks[assigned_tracks['id'] > 0]  # Tracks pertenecientes a estimated_tracks
    right_distances = opt_distances[ground_truth['opt_track_id'].unique() > 0]

    # Parámetros de desempeño:
    alpha = 1 - opt_distance/max_distance
    beta = (max_distance - opt_distance)/(max_distance + wrong_max_distance)
    # number non dummy tracks assigned to ground_truth tracks
    TP = len(right_tracks['id'].unique())
    # number of dummy tracks assigned to ground_truth tracks:
    FN = len(assigned_tracks[assigned_tracks['id'] <= 0]['id'].unique())
    # number of non dummy tracks not assigned to ground_truth tracks"
    FP = len(wrong_tracks['id'].unique())
    JSC = TP/(TP + FN + FP)

    rmse = np.sqrt(np.mean(right_distances ** 2))
    min_error = np.min(right_distances)
    max_error = np.max(right_distances)
    sd = np.std(right_distances)

    # Number of right positions in tracks assigned to ground truth tracks.
    TP_positions = 0
    # Number of positions assigned to dummy tracks:
    FN_positions = ground_truth[ground_truth['opt_track_id'] <= 0].shape[0]
    # Number of positions of tracks not assigned to ground truth tracks:
    FP_positions = wrong_tracks.shape[0]

    right_tracks_grouped = right_tracks.groupby('id')
    gt_grouped = ground_truth.groupby('id')

    for track_id in right_tracks['id'].unique():
        right_track = right_tracks_grouped.get_group(track_id)
        gt_id = right_track['opt_track_id'].unique()[0]
        gt_track = gt_grouped.get_group(gt_id)

        num_frames = max(right_track['frame'].to_numpy(dtype='int32').max(),
                         gt_track['frame'].to_numpy(dtype='int32').max())+1
        min_frame = min(right_track['frame'].to_numpy(dtype='int32').min(),
                        gt_track['frame'].to_numpy(dtype='int32').min())

        max_x = max(right_track['x'].to_numpy(dtype='int32').max(), gt_track['x'].to_numpy(dtype='int32').max())
        max_y = max(right_track['y'].to_numpy(dtype='int32').max(), gt_track['x'].to_numpy(dtype='int32').max())

        tracks_right_np, _ = pandas_tracks_to_numpy(right_track, num_frames, max_x*max_y)
        tracks_gt_np, _ = pandas_tracks_to_numpy(gt_track, num_frames, max_x*max_y)

        distances = np.linalg.norm((tracks_right_np[:, :, 0] - tracks_gt_np[:, :, 0]), axis=0)
        TP_positions += np.sum(distances < max_dist) - min_frame
        FN_positions += np.sum(tracks_gt_np[0, distances > max_dist, 0] < max_x*max_y)
        FP_positions += np.sum(tracks_right_np[0, distances > max_dist, 0] < max_x * max_y)

    JSC_positions = TP_positions/(TP_positions + FN_positions + FP_positions)

    performance_measures = {
        'alpha': alpha,
        'beta': beta,
        'TP Tracks': TP,
        'FN Tracks': FN,
        'FP Tracks': FP,
        'JSC Tracks': JSC,
        'RMSE': rmse,
        'Min': min_error,
        'Max': max_error,
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
tracks_csv = pd.read_csv('Spots in tracks statistics-dataset1-10Hz.csv')
# print(tracks_csv.columns)
tracks_csv.rename(columns={'TRACK_ID': 'id',
                           'POSITION_X': 'x',
                           'POSITION_Y': 'y',
                           'FRAME': 'frame'}, inplace=True)
gt_tracks = pd.read_csv('dataset_1_data.csv')
# print(gt_tracks.columns)
gt_tracks.rename(columns={'id_particle': 'id'}, inplace=True)
gt_tracks = gt_tracks[gt_tracks['frame'] > 1]
print('---------------------------------------------------TrackMate---------------------------------------------------')
# print('tracks: \n', 'shape:', tracks_csv.shape, '\n', tracks_csv.head())
# print('----------------------------------------------------')
# print('gt: \n', 'shape:', gt_tracks.shape, '\n', gt_tracks.head())
# print('----------------------------------------------------')
start = time.time()
error = track_set_error(gt_tracks, tracks_csv, 40)
end = time.time()
print('Time to run track_set_error: ', end - start)
print('\n Performance Measures:')
PrettyPrinter(sort_dicts=False).pprint(error)

print('---------------------------------------------------ENN JPDAF---------------------------------------------------')
tracks_csv = pd.read_csv('tracks_enn_jpdaf.csv')
# print(tracks_csv.columns)
tracks_csv = tracks_csv[tracks_csv['frame'] < 40]

t0 = time.time()
error = track_set_error(gt_tracks, tracks_csv, 40)
t1 = time.time()
print('TIme to run track_set_error: ', t1-t0)
print('\n Performance Measures:')
PrettyPrinter(sort_dicts=False).pprint(error)

print('---------------------------------------------------JPDAF C++---------------------------------------------------')
tracks_csv = pd.read_csv('dataset_1_10Hz_C.csv')
# print(tracks_csv.columns)
tracks_csv.rename(columns={'track_id': 'id'}, inplace=True)
tracks_csv = tracks_csv[tracks_csv['frame'] < 40]

t0 = time.time()
error = track_set_error(gt_tracks, tracks_csv, 40)
t1 = time.time()
print('TIme to run track_set_error: ', t1-t0)
print('\n Performance Measures:')
PrettyPrinter(sort_dicts=False).pprint(error)
# ----------------------------------------------------------------------------------------------------------------------
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
