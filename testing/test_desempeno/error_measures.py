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
    '''
    Parametros:
        ground_truth_df (df(id, x, y, total_pixels, mask, frame)): Coordenadas ground truth de las partículas.
        df_Y (df(id, x, y, total_pixels, mask, frame)): Coordenadas medidas de las partículas.
        M (int): Ancho de la imagen.
        N (int): Largo de la imagen.
        max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.

    Returwn:
        TP (int): Verdaderos Positivos
        FN (int): Falsos Negativos
        FP (int): Falsos Positivos
        JSC (int): Índice de Jaccard
    '''

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





'''
PRUEBA: 

N = 15
F = 5

X = np.zeros((N*F,3))
Y = np.zeros((N*F,3))


for n in range(N):
	for f in range(F):
		X[n*F+f,:] = [250+np.random.normal(0, 20), 250+np.random.normal(0, 20), f]
		Y[n*F+f,:] = [500+np.random.normal(0, 20), 500+np.random.normal(0, 20), f]

for i in range(30):
	Y[i,:] = X[i,:]
df_X = pd.DataFrame(X,columns = ['x','y','frame'])
df_Y = pd.DataFrame(Y,columns = ['x','y','frame'])

#print(df_X)
#print(df_Y)

TP, FN, FP, JSC = error_measures(df_X, df_Y, 512, 512, 30)
print(TP, FN, FP, JSC)
'''
