# Joint Probabilistic Data Association Tracking (JPDAFTracker)
JPDAFTracker is a tracker based on joint probabilistic data association filtering.

<p align="center">
<a href="https://www.youtube.com/watch?v=KlXpaKh8hDY"  target="_blank"><img src="https://img.youtube.com/vi/KlXpaKh8hDY/0.jpg"/></a>
</p>
<br>

## Requirements
Esto hay que instalarlo a parte. Yo usé la linea de comandos ("brew install opencv" que es el análogo a "sudo install opencv")
Revisen que sea el opencv 4, para que les ande este código que cambié.

Es necesario que instalen cmake antes de opencv. Les dejo el link, para que puedan una vez que descargaron el cmake, agregarlo al PATH de windows. <a href="https://docs.alfresco.com/4.2/tasks/fot-addpath.html" > link </a >

* <a href="https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html"> OpenCV </a>
Eigen es el mismo procedimiento que opencv 
* Eigen

## How to build

JPDAFTracker works under Linux environments. I recommend a so-called out of source build which can be achieved by the following command sequence:

* mkdir build
* cd build
* cmake ../
* make -j<number-of-cores+1>
(La últmia línea es "make -j 4" por ejemplo)

## Params
```bash
[PD] #DETECTION PROBABILITY
1

[PG] #GATE PROBABILITY
0.4

[LOCAL_GSIGMA] #THRESHOLD OF GATING
15

[LOCAL_ASSOCIATION_COST] #ASSOCIATION COSTS
40

[GLOBAL_GSIGMA] #THRESHOLD OF GATING
0.1

[GLOBAL_ASSOCIATION_COST] #ASSOCIATION COSTS
50

[LAMBDA] #CONSTANT
2

[GAMMA] #G1,G2 INITIAL COVARIANCE P MATRIX
10 10

[R_MATRIX] #2x2 MEASUREMENT NOISE COVARIANCE MATRIX
100 0
0 100

[DT] #dt
0.4

[MIN_ACCPETANCE_RATE] #min rate for convalidating a track
10

[MAX_MISSED_RATE] #max rate for deleting a track
9
```

## How to use

En la línea de comandos
```bash
./jpdaf_tracker ../config/params.txt /path/to/the/detection_file.txt /path/to/the/image_folder 
```
