from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm


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
    tracks_new[:, tracks_grouped.get_group(ids[0])['frame'].to_numpy(dtype='int32'), 0] = 1
    for i in range(num_tracks):
        if ~(np.isnan(tracks_grouped.get_group(ids[i])[['x', 'y']].to_numpy())).any():
            tracks_new[:, tracks_grouped.get_group(ids[i])['frame'].to_numpy(dtype='int32'), i] = \
                tracks_grouped.get_group(ids[i])[['x', 'y']].to_numpy().T

    return tracks_new, ids


def get_optimal_track_assignment(tracks_a, tracks_b, max_dist):
    """
    Determina el subconjunto Y_opt de Y que cumple dist(X,Y_opt) = min(X, Y*), siendo Y* cualquier subconjunto de Y.
    Parameters:
        tracks_a (pd.DataFrame): Con las columnas ['id', 'x', 'y', 'frame'].
        tracks_b (pd.DataFrame): Columnas ['id', 'x', 'y', 'frame'].
        max_dist (float): Máxima distancia entre dos partículas para considerar que no son la misma.
                        max_dist debería ser del órden del doble del tamaño promedio de las partículas.
    Returns:
        tracks_a (DataFrame): Se agrega la columna 'opt_track_id' al dataframe de entrada tracks_a, indicando el
                                id_track de track_b asignado.
        tracks_b (DataFrame): Se agrega la columna 'opt_track_id' al dataframe de entrada tracks_b, indicando el
                                id_track de track_a asignado.
        cost (list): Distancias de las trayectorias asignadas. Ordenado ...
    """
    num_frames = int(max(np.nanmax(tracks_a['frame']), np.nanmax(tracks_b['frame'])) + 1)
    max_x = max(tracks_a['x'].to_numpy(dtype='int32').max(), tracks_b['x'].to_numpy(dtype='int32').max())
    max_y = max(tracks_a['y'].to_numpy(dtype='int32').max(), tracks_b['x'].to_numpy(dtype='int32').max())

    tracks_a_np, ids_a = pandas_tracks_to_numpy(tracks_a, num_frames, max_x*max_y)
    tracks_b_np, ids_b = pandas_tracks_to_numpy(tracks_b, num_frames, max_x*max_y)
    num_tracks_a = len(ids_a)
    num_tracks_b = len(ids_b)

    cost = np.zeros([num_tracks_a, num_tracks_b])
    for i in tqdm(range(num_tracks_a), desc='ground truth tracks'):
        for j in range(num_tracks_b):
            distances = np.linalg.norm((tracks_a_np[:, :, i] - tracks_b_np[:, :, j]), axis=0)
            cost[i, j] = np.sum(np.minimum(distances, max_dist))
    # print(cost)
    row_ind, col_ind = linear_sum_assignment(cost)

    tracks_a_new = tracks_a.copy()
    tracks_b_new = tracks_b.copy()

    for i in range(len(col_ind)):
        tracks_a_new.at[tracks_a['id'] == ids_a[row_ind[i]], 'opt_track_id'] = ids_b[col_ind[i]]
        tracks_b_new.at[tracks_b['id'] == ids_b[col_ind[i]], 'opt_track_id'] = ids_a[row_ind[i]]

    return tracks_a_new, tracks_b_new, cost[row_ind, col_ind]


def get_optimal_position_assignment(gt_positions, est_positions, c, p, p_prime, alpha):
    """
    Calcula la asignación óptima para las posiciones en un determinado frame.
    Se calculan los costos de la matriz con np.array's.

    Parameters:
        gt_positions (pd.DataFrame): Posiciones de ground truth en determinado frame.
        est_positions (pd.DataFrame): Posiciones estimadas a evaluar en determinado frame.
        c (float):  cut-off parameter, a measure of penalty assigned to missed or false tracks.
        p (float): 1 ≤ p < ∞ is the OSPA metric order parameter.
        p_prime (int):  1 ≤ p′ < ∞ is the base distance order parameter.
        alpha (float): ∈ [0, c] in controls the penalty assigned to the labeling error.
    Returns:
        sum_base_dists (float): Suma de las distancias obtenidas de la asignación óptima.
    """
    max_x = max(gt_positions['x'].to_numpy(dtype='int32').max(), est_positions['x'].to_numpy(dtype='int32').max())
    max_y = max(gt_positions['y'].to_numpy(dtype='int32').max(), est_positions['x'].to_numpy(dtype='int32').max())
    gt_positions_aux = gt_positions.copy()
    est_positions_aux = est_positions.copy()
    gt_positions_aux['frame'] = 0
    est_positions_aux['frame'] = 0
    gt_positions_np, gt_ids = pandas_tracks_to_numpy(gt_positions_aux, 1, max_x*max_y)
    est_positions_np, est_ids = pandas_tracks_to_numpy(est_positions_aux, 1, max_x*max_y)

    cost_matrix = np.ones((len(gt_ids), len(est_ids)))*c
    for gt_id in range(len(gt_ids)):
        gt_position = gt_positions_np[:, 0, gt_id]
        gt_opt_id = gt_positions[gt_positions['id'] == gt_ids[gt_id]]['opt_track_id'].unique()
        for est_id in range(len(est_ids)):
            est_position = est_positions_np[:, 0, est_id]
            localisation_base_dist = np.linalg.norm(gt_position - est_position, ord=p_prime)
            labeling_error = alpha * (not est_ids[est_id] == gt_opt_id)
            base_dist = (localisation_base_dist ** p_prime + labeling_error ** p_prime) ** (1/p_prime)

            cost_matrix[gt_id, est_id] = min(c, base_dist) ** p
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sum_base_dists = np.sum(cost_matrix[row_ind, col_ind])

    return sum_base_dists


def ospa_multiprocessing_aux(positions):
    """
    Función auxiliar para paralelizar en frames el calculo de la distancia ospa.
    Parameters:
        positions (tuple): de la forma (est_positions, gt_positions, c, p, p_prime, alpha).
    Returns:
        ospa_dist (float): Distancia ospa para un frame.
    """
    est_positions, gt_positions, c, p, p_prime, alpha = positions

    if len(gt_positions['id'].unique()) <= len(est_positions['id'].unique()):
        d_positions = est_positions
        x_positions = gt_positions
    else:
        d_positions = gt_positions
        x_positions = est_positions

    m = len(x_positions['id'].unique())
    n = len(d_positions['id'].unique())
    sum_base_dists = get_optimal_position_assignment(x_positions, d_positions, c, p, p_prime, alpha)
    ospa_dist = ((1 / n) * (sum_base_dists ** p + (n - m) * (c ** p))) ** (1 / p)

    return ospa_dist


def ospa_distance(ground_truth, estimated_tracks, c, p, p_prime, alpha):

    """
    Calcula la distancia ospa entre conjuntos de trayectorias, toma como entrada dos conjuntos de
    trayectorias con sus respectivas asignaciones óptimas.
    Basado en el paper:
    Ristic, B., Vo, B-N., Clark, D., & Vo, B-T. (2011). A Metric for Performance Evaluation
    of Multi-Target Tracking Algorithms. IEEE Transactions on Signal Processing, 59(7), 3452-3457.
    https://doi.org/10.1109/TSP.2011.2140111

    Parameters:
        ground_truth (pd.DataFrame): contiene las columnas (id, x, y, frame, opt_track_id).
        estimated_tracks (pd.DataFrame): contiene las columnas (id, x, y, frame, opt_track_id).
        c (float):  cut-off parameter, a measure of penalty assigned to missed or false tracks.
        p (float): 1 ≤ p < ∞ is the OSPA metric order parameter.
        p_prime (int):  1 ≤ p′ < ∞ is the base distance order parameter.
        alpha (float): ∈ [0, c] in controls the penalty assigned to the labeling error.

    Returns:
        ospa_dist (float): Optimal Sub-Pattern Assignment distance.
    """

    frames = ground_truth['frame'].unique()
    gt_groups = ground_truth.groupby('frame')
    est_groups = estimated_tracks.groupby('frame')
    list_tuples = []

    ospa = 0
    for frame_num in frames:
        if frame_num in estimated_tracks['frame'].unique():
            list_tuples.append((est_groups.get_group(frame_num), gt_groups.get_group(frame_num), c, p, p_prime, alpha))
        else:
            ospa += 2*c
    pool = multiprocessing.Pool()
    ospa_dists = pool.map(ospa_multiprocessing_aux, list_tuples)
    ospa += np.sum(ospa_dists)

    return ospa


def track_set_error(ground_truth, estimated_tracks, max_dist):
    """
    Toma como entrada del conjunto de trayectorias a evaluar y el ground truth con que comparar.
    Parameters:
        ground_truth (pd.DataFrame): contiene las columnas (id, x, y, frame).
        estimated_tracks (pd.DataFrame): contiene las columnas (id, x, y, frame).
        max_dist (float): distancia máxima, si la distancia entre dos puntos es mayor a la distancia máxima se considera un
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
    print('Computing optimal track assignment...')
    t0 = time.time()
    ground_truth, tracks_extended, opt_distances = get_optimal_track_assignment(ground_truth, tracks_extended, max_dist)
    t1 = time.time()
    print('Time to run optimal assignment: {:.2f}s'.format(t1 - t0))
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

    t0 = time.time()
    print('Computing ospa distance...')
    ospa = ospa_distance(ground_truth, tracks_extended[tracks_extended['id'] > 0],
                         c=max_dist, p=0.9, p_prime=2, alpha=max_dist)
    t1 = time.time()
    print('Time to run ospa: {:.2f}s'.format(t1 - t0))

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
        'JSC Positions': JSC_positions,
        'OSPA': ospa
    }
    return performance_measures
