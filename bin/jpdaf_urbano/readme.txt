VideoSegmentation_rev_11L.m:
  Realiza la detección de las partículas.
  Toma como entrada la secuencia a ser estudiada en forma de video, hay que especificar su ubicación.
  La salida es un .dat con las detecciones.
  Las Primeras dos filas del archivo son las pocisiones de cada partícula detectada y la tercera fila es el frame correspondiente a cada detección.
 
VideoSpermTracker_rev_26L60sec.m:
  Realiza el tracking de las partículas.
  Toma como entrada la secuencia a ser estudiada en forma de video y las detecciones de esa secuencia cuyo formato es el de la salida de "VideoSegmentation_rev_11L.m ".
  