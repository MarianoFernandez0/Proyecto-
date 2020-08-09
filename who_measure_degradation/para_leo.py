import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

image = np.zeros((100, 100))


pos = np.array([[5,5],[10,12],[22,25],[33,40],[40,15],[50,3],[65,13],[72,23]])
posX = [5,0,22,33,40,50,65,72]
posY = [5,12,25,40,15,3,13,23]


def avgPath(X, Y):
    NumberNewPoints = 5
    xPath = []
    yPath = []

    for i in range(len(X) - 1):
        xvals = np.linspace(X[i], X[i + 1], NumberNewPoints + 2)  # crear 5 puntos en el medio (7 en total)
        yvals = np.linspace(Y[i], Y[i + 1], NumberNewPoints + 2)
        if i != 0:
            xvals = xvals[1:-1]  # saco el primer y ultimo lugar
            yvals = yvals[1:-1]
        for j in range(len(xvals)):
            # obtengo caminos con muchos puntos intermedios (5 entre cada dos puntos originales)
            xPath.append(xvals[j])
            yPath.append(yvals[j])

    windowSize = NumberNewPoints * 2
    xPathSmooth = []  # se promedia con una ventana - al final se obtiene un camino con #windowSize puntos menos
    yPathSmooth = []  # la ventana movil se hace desde 0+windowsize/2 hasta fin-windowsize/2
    for i in range(math.trunc(windowSize / 2), int(len(xPath) - math.ceil(windowSize / 2))):
        xPathSmooth.append(np.mean(xPath[int(i - math.trunc(windowSize / 2)):int(i + math.ceil(windowSize / 2))]))
        yPathSmooth.append(np.mean(yPath[int(i - math.trunc(windowSize / 2)):int(i + math.ceil(windowSize / 2))]))

        # uno el primer y ultimo punto con el avg path asi me queda una avg path completo

    xvals1 = np.linspace(X[0], xPathSmooth[0], math.ceil(windowSize / 2) + 2)  # se completa la parte del smoothpath que
    yvals1 = np.linspace(Y[0], yPathSmooth[0],
                         math.ceil(windowSize / 2) + 2)  # falta entre el primer valor y el primero
    xvalsEnd = np.linspace(xPathSmooth[-1], X[-1], math.ceil(windowSize / 2) + 2)  # del array smooth
    yvalsEnd = np.linspace(yPathSmooth[-1], Y[-1], math.ceil(windowSize / 2) + 2)
    for i in range(len(xvals1) - 1):
        xPathSmooth = [xvals1[-i - 2]] + xPathSmooth + [xvalsEnd[i + 1]]
        yPathSmooth = [yvals1[-i - 2]] + yPathSmooth + [yvalsEnd[i + 1]]

    return xPathSmooth, yPathSmooth

avgX, avgY = avgPath(posX,posY)

plt.plot([p[0] for p in pos], [p[1] for p in pos])
plt.plot(avgX,avgY)
plt.show()
