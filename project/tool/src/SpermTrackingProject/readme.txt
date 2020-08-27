VideoSegmentation_rev_11L.m:
  Realiza la detección de las partículas.
  Toma como entrada la secuencia a ser estudiada en forma de video, hay que especificar su ubicación.
  La salida es un .dat con las detecciones.
  Las Primeras dos filas del archivo son las pocisiones de cada partícula detectada y la tercera fila es el frame correspondiente a cada detección.
 
VideoSpermTracker_rev_26L60sec.m:
  Realiza el tracking de las partículas.
  Toma como entrada la secuencia a ser estudiada en forma de video y las detecciones de esa secuencia cuyo formato es el de la salida de "VideoSegmentation_rev_11L.m ".
  Genera un .csv con las trayectorias detectadas.
  Parámetros:
  	
  	dataFile: ruta al archivo con las detecciones.
	videoFile: ruta al video.
	csv_tracks: ruta a donde debería guardarse el csv con las trayectorias.
	
	mttAlgorithm: algoritmo para realizar el trackink, opciones posibles: % 1 = NN, GNN, PDAF, JPDAF, ENN-JPDAF, Iterated Multi-assignment.
	plotResults: 1 para mostrar los resultados del tracking cuadro por cuadro.
	saveMovie: 1 para guardar un video con el resultado.
	snapShot: 1 para guardar un snapshot.
	plotTrackResults: 1 para plotear los resultados finales del tracking.
	analyzeMotility: 1 para obtener las medidas de motilidad.
	
	px2um: escala de pixel a micrometro.
	ROIx: región de interés en x.
	ROIy: región de interés en y.
	T: tiempo de un frame.
	numFrames: número de frames de la secuencia.
	
	
	N: matriz de covarianza del ruido. En el paper asumen N(k) = sigma_n^2I_2, sigma_n = 2 um.
	Q0: inicializaciOn de la matriz de covarianza del ruido del proceso (CWNA Process Noise)
	Q0 = qMat * qIntensity
	deltaV:
	qIntensity: en el paper \tilde{q_0} = 20 um/s


	% Statistical Parameters
	PG: Prob. that a detected target falls in validation gate (Paper: PG = 0.997)
	PD: Prob. of detection

	lam_f: Expected number of measurements due to clutter per unit area of the surveillance space per scan of data.
	lam_n: valor esperado de la cantidad de nuevas medidas correspondientes a nuevas tracks por unidad de área, "per scan of data" 

	Pdelete: Probabilidad de eliminar una track verdadera.
	Pconfirm: Probabilidad de confiermar una track falsa.

	gx: restricción de posición, se cumple cuando d^2<gx. En el paper: gx = chi2inv(PG, 2).
	gv: restricción de posición, se cumple cuando v/T<gv. En el paper: gv = 300 um/s.
	
  	
