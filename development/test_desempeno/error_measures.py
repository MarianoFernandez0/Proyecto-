from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd


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

    min_frame_a = track_a['frame'].min()
    min_frame_b = track_b['frame'].min()
    min_frame = int(min(min_frame_a, min_frame_b))

    max_frame_a = track_a['frame'].max()
    max_frame_b = track_b['frame'].max()
    max_frame = int(max(max_frame_a, max_frame_b))

    distance = 0
    for frame in range(min_frame, max_frame+1):
        if frame < min_frame_a:
            distance += max_dist
        elif frame < min_frame_b:
            distance += max_dist
        elif frame > max_frame_a:
            distance += max_dist
        elif frame > max_frame_b:
            distance += max_dist
        else:
            coord_track_a = track_a[track_a['frame'] == frame][['x', 'y']].to_numpy().squeeze()
            coord_track_b = track_b[track_b['frame'] == frame][['x', 'y']].to_numpy().squeeze()
            l2_distance = np.sqrt((coord_track_a[0] - coord_track_b[0]) ** 2 +
                                  (coord_track_a[1] - coord_track_b[1]) ** 2)
            distance += min(l2_distance, max_dist)

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
        tracks_a: df con las columnas ['id', 'x', 'y', 'frame'].
        tracks_b: df con las columnas ['id', 'x', 'y', 'frame'].
        max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.
                        max_dist debería ser del órden del doble del tamaño promedio de las partículas.
    Returns:
        tracks_b_opt: Subconjunto de Y que minimiza la distancia entre X e tracks_Y_opt.
    """
    ids_a = tracks_a['id'].unique()
    ids_b = tracks_b['id'].unique()
    num_tracks_a = len(ids_a)
    num_tracks_b = len(ids_b)

    cost = np.zeros([num_tracks_a, num_tracks_b])
    for i in range(num_tracks_a):
        for j in range(num_tracks_b):
            cost[i, j] = distance_between_two_tracks(tracks_a[tracks_a['id'] == ids_a[i]],
                                                     tracks_b[tracks_b['id'] == ids_b[j]],
                                                     max_dist)

    row_ind, col_ind = linear_sum_assignment(cost)
    dist = cost[row_ind, col_ind].sum()

    tracks_b_opt = pd.DataFrame(columns=tracks_b.columns)
    for i in range(len(col_ind)):
        tracks_b_opt = pd.concat([tracks_b_opt, tracks_b[tracks_b['id'] == ids_b[col_ind[i]]]])

    return tracks_b_opt, dist


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


def track_set_error(ground_truth, tracks, max_dist):
    """
    Toma como entrada del conjunto de trayectorias a evaluar y el ground truth con que comparar.
    Parameters:
        ground_truth: dataframe que contiene las columnas (id, x, y, frame).
        tracks: dataframe que contiene las columnas (id, x, y, frame).
        max_dist: distancia máxima, si la distancia entre dos puntos es mayor a la distancia máxima se considera un
                    error en la asignacion.
    Returns:
        alpha (float): Definido en "Performance measures". Entre 0 y 1, es 1 si los conjuntos son iguales, y 0 en el
                        mayor error posible.
        beta (float): Definido en "Performance measures". Entre 0 y alpha, es alpha si no hay tracks erroneas y converge
                        a cero a medida que el número aumenta.
    """

    dummy_tracks = {'id': -ground_truth['id'].unique()}
    dummy_tracks = pd.DataFrame(data=dummy_tracks, columns=['id', 'frame'])

    tracks_extended = pd.concat([tracks, dummy_tracks])

    tracks_opt, distance = get_optimal_track_assignment(ground_truth, tracks_extended, max_dist)
    _, max_distance = get_optimal_track_assignment(ground_truth, dummy_tracks, max_dist)

    discarded_tracks = tracks[~tracks.index.isin(tracks_opt.index)]            # tracks - tracks_opt
    discarded_dummy_tracks = {'id': -discarded_tracks['id'].unique()}
    discarded_dummy_tracks = pd.DataFrame(data=discarded_dummy_tracks, columns=['id', 'frame'])
    print('dummy', discarded_dummy_tracks.head())
    _, discarded_to_max_distance = get_optimal_track_assignment(discarded_tracks, discarded_dummy_tracks, max_dist)

    print('distance: ', distance)
    print('max_dist: ', max_distance)
    print('discarded_to_max_distance: ', discarded_to_max_distance)
    alpha = 1 - distance/max_distance
    beta = (max_distance - distance)/(max_distance + discarded_to_max_distance)
    return alpha, beta
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
tracks_csv = pd.read_csv('tracks.csv')
gt_tracks = pd.read_csv('_data.csv')
tracks_csv = tracks_csv[tracks_csv['id'] < 5]
print('tracks: \n', tracks_csv.head())
print('gt: \n', gt_tracks.head())
alpha, beta = track_set_error(tracks_csv, tracks_csv, 10)
print('alpha: ', alpha)
print('beta: ', beta)

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
