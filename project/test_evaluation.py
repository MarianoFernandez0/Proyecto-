import tifffile
import time
from evaluation import evaluation
import numpy as np
from ground_truth.generador_simulaciones_2 import generate_sequence
from test_desempeno.error_measures import error_measures, fix_particles_oustide


poblaciones = []
mean = np.array([20.7247332, 9.61818939])
cov = np.array([[103.80124818, 21.61793687],
				 [ 21.61793687, 14.59060681]])
vm = 3
poblacion = {
	'particles' : 25,
	'mean' : mean,
	'cov' : cov,
	'mean_velocity' : vm,
	'sigma_v' : vm * 0.1,
	'sigma_theta' : 10
}
poblaciones.append(poblacion)

mean = np.array([15, 6])
cov = np.array([[103.80124818, 21.61793687],
[ 21.61793687, 14.59060681]])
vm = 5

poblacion = {
	'particles' : 50,
	'mean' : mean,
	'cov' : cov,
	'mean_velocity' : vm,
	'sigma_v' : vm * 0.2,
	'sigma_theta' : 15
}

poblaciones.append(poblacion)

print('Now generating ground truth sequence')
t0 = time.time()

df_ground_truth = generate_sequence(M = 512, N = 512, frames = 2, sigma_r = 4, poblaciones = poblaciones)

df_ground_truth_filtered = df_ground_truth[df_ground_truth.intensity > 50] 

#df_ground_truth_filtered = fix_particles_oustide(df_ground_truth_filtered, M = 512, N = 512)

t1 = time.time()
print('Finished running df_ground_truth in:', t1-t0)	

tif = tifffile.TiffFile('ground_truth/output/salida.tif')

print('Now detecting particles in the sequence')
t0 = time.time()

df_detected = evaluation(tif)
t1 = time.time()
print('Finished running evaluation in:', t1-t0)	

print('Now computing the error measures')
t0 = time.time()

TP, FN, FP, JSC = error_measures(df_X=df_ground_truth, df_Y=df_detected, M=512, N=512, max_dist=60)
t1 = time.time()
print('Finished running error_measures in:', t1-t0)	

print('True Positives =' ,TP)
print('False Negatives = ', FN)
print('False Positives = ', FP)
print('JSC =', JSC)

print('df_ground_truth_filtered:')
print(df_ground_truth_filtered)
print('df_detected:')
print(df_detected)