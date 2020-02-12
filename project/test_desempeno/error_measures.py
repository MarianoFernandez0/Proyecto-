from scipy.optimize import linear_sum_assignment
import numpy as np

def fix_particles_oustide(X,M,N):
	
	X_fix = []								#Se guarda en esta lista las partículas que están dentro del rango posible
	for p in X:
		if (p[0] > 0 and p[0] < M and p[1] >0 and p[1] < N):
			X_fix.append(p)
	X_fix = np.array(X_fix)

	return X_fix

def get_optimal_assignment(X,Y):
	
	cost = np.zeros([len(X),len(X)])    
	for i in range(len(X)):
		for j in range(len(Y)):
			cost[i,j] = np.linalg.norm(X[i]-Y[j],2)

	row_ind, col_ind = linear_sum_assignment(cost)

	Y_opt = np.zeros(Y.shape)
	for i in range(len(Y)):
		Y_opt[i] = Y[col_ind[i]]

	return Y_opt


def error_measures(X, Y, M, N, max_dist):
	'''
	Parametros:
		X (numpy.array((particles,frames))): Coordenadas ground truth de las partículas.
		Y (numpy.array((particles,frames))): Coordenadas medidas de las partículas.
		M (int): Ancho de la imagen.
		N (int): Largo de la imagen.
		max_dist (int): Máxima distancia entre dos partículas para considerar que no son la misma.

	Return:
		TP (int): Verdaderos Positivos
		FN (int): Falsos Negativos
		FP (int): Falsos Positivos
		JSC (int): Índice de Jaccard
	'''

	X_fix = fix_particles_oustide(X,M,N)
	
	total_frames = X.shape[1]

	for f in range(total_frames):
		X_f = X[:,f]
		Y_f = Y[:,f]
		Y_aux = np.concatenate(Y_f, max_dist*np.ones(len(X_f)))
		Y_opt = get_optimal_assignment(X_f, Y_aux)

		TP += len(Y_opt[Y_opt < max_dist]) 
		FN += len(Y_opt[Y_opt == max_dist])
		FP += len(Y_opt) - len(Y_opt[Y_opt!=max_dist])

	JSC = TP/(TP + FN + FP)

	return TP, FN, FP, JSCb

N = 50
F = 20

X = np.zeros((N,F))

for n in range(N):
	for f in arange(F):
		X[n,f] = 250*np.random.normal(0, 1, 2)