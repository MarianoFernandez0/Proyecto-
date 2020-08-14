import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

image = np.zeros((100, 100))


pos = np.array([[5,5],[10,12],[22,25],[33,40],[40,15],[50,3],[65,13],[72,23]])
posX = [38.65984654731458,37.17821782178218,32.7843137254902,32.241379310344826,31.94638069705093,27.0,26.40921409214092,26.54395604395605,22.159340659340657,20.13440860215054,21.400537634408604,17.474474474474466,14.38208955223881,15.786833855799367,12.985507246376807,8.901898734177216,9.413173652694613,8.319018404907975,6.083682008368201,6.000000000000001,5.436426116838488,4.075949367088608,3.691275167785235,3.664772727272727,1.333333333333333,-0.6541570971664625,-1.8153120941722105]
posY = [359.23273657289,356.87871287128706,359.97794117647067,356.28571428571416,352.9356568364612,355.49999999999994,353.4552845528455,348.49450549450546,350.64835164835165,349.92473118279565,344.61290322580635,345.7807807807808,346.2447761194029,341.1128526645768,340.97971014492754,342.1297468354431,337.12574850299393,336.0184049079756,338.15899581589963,334.01067615658366,330.39862542955325,334.1202531645569,332.39597315436237,328.88068181818176,330.6666666666667,327.6962916688727,326.49184323777394]

def avgPath(X, Y):
    NumberNewPoints = int(np.round(len(posX)/5))
    xPath = []
    yPath = []

    for i in range(len(X) - 1):
        xvals = np.linspace(X[i], X[i + 1], NumberNewPoints + 2)  # crear 5 puntos en el medio (7 en total)
        yvals = np.linspace(Y[i], Y[i + 1], NumberNewPoints + 2)

        for j in range(len(xvals)):
            # obtengo caminos con muchos puntos intermedios (5 entre cada dos puntos originales)
            xPath.append(xvals[j])
            yPath.append(yvals[j])

    windowSize = NumberNewPoints * 5
    xPathSmooth = [np.mean(xPath[i:i + windowSize]) for i in range(0, len(xPath) - windowSize)]
    yPathSmooth = [np.mean(yPath[i:i + windowSize]) for i in range(0, len(yPath) - windowSize)]

    xvals1 = np.linspace(X[0], xPathSmooth[0], windowSize)  # se completa la parte del smoothpath que
    yvals1 = np.linspace(Y[0], yPathSmooth[0], windowSize)  # falta entre el primer valor y el primero
    xvalsEnd = np.linspace(xPathSmooth[-1], X[-1], windowSize)  # del array smooth
    yvalsEnd = np.linspace(yPathSmooth[-1], Y[-1], windowSize)
    for i in range(len(xvals1) - 1):
        xPathSmooth = [xvals1[-i - 2]] + xPathSmooth + [xvalsEnd[i + 1]]
        yPathSmooth = [yvals1[-i - 2]] + yPathSmooth + [yvalsEnd[i + 1]]

    return xPathSmooth, yPathSmooth


def avgPathorg(X, Y):
    NumberNewPoints = 10
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

    windowSize = NumberNewPoints * 5
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

plt.plot(posX, posY, marker='o')
for i in range(len(posX)):
    plt.text(posX[i], posY[i], i)
plt.plot(avgX,avgY, marker='*')
plt.show()
