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


def get_optimal_assignment(X, Y, max_dist):
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


def error_measures(ground_truth_df, detected_df, max_dist):
    """
    Parameters:
        ground_truth_df (df(id, x, y, total_pixels, mask, frame)): Coordenadas ground truth de las partículas.
        df_Y (df(id, x, y, total_pixels, mask, frame)): Coordenadas medidas de las partículas.
        M (int): Ancho de la imagen.
        N (int): Largo de la imagen.
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

    min_frame_a = track_a['frame'].min()
    min_frame_b = track_b['frame'].min()
    min_frame = int(min(min_frame_a, min_frame_b))

    max_frame_a = track_a['frame'].max()
    max_frame_b = track_b['frame'].max()
    max_frame = int(max(max_frame_a, max_frame_b))

    distance = 0
    for frame in range(min_frame, max_frame):
        if frame < min_frame_a:
            distance += max_dist
        elif frame < min_frame_b:
            distance += max_dist
        elif frame > max_frame_a:
            distance += max_dist
        elif frame > max_frame_b:
            distance += max_dist
        else:
            coord_track_a = track_a[track_a['frame'] == frame][['x', 'y']].squeeze()
            coord_track_b = track_b[track_b['frame'] == frame][['x', 'y']].squeeze()
            l2_distance = np.sqrt((coord_track_a[0] - coord_track_b[0]) ** 2 +
                                  (coord_track_a[1] - coord_track_b[1]) ** 2)
            distance += max(l2_distance, max_dist)

    return distance


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
