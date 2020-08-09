import numpy as np
import math


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

    windowSize = NumberNewPoints * 3
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


def VSL(X, Y, T):
    dist = math.sqrt((X[-1] - X[0]) ** 2 + (Y[-1] - Y[0]) ** 2)
    time = T[-1] - T[0]
    vsl = dist / time
    return vsl


def VCL(X, Y, T):
    vel = []
    for i in range(len(X) - 1):
        dist = math.sqrt((X[i + 1] - X[i]) ** 2 + (Y[i + 1] - Y[i]) ** 2)
        time = T[i + 1] - T[i]
        vel.append(dist / time)
    vcl = np.mean(vel)
    return vcl


def VAP(X, Y, avgPathX, avgPathY, T):
    vel = []
    minIndexOld = 0
    for j in range(1, len(X)):
        minDist = float('Inf')
        for i in range(len(avgPathX)):
            dist = math.sqrt((X[j] - avgPathX[i]) ** 2 + (Y[j] - avgPathY[i]) ** 2)
            if dist < minDist:
                minIndex = i
                minDist = dist
        dist = 0
        if minIndex >= minIndexOld:
            for i in range(minIndexOld, minIndex):
                dist = dist + math.sqrt((avgPathX[i + 1] - avgPathX[i]) ** 2 + (avgPathY[i + 1] - avgPathY[i]) ** 2)
        else:
            for i in range(minIndex, minIndexOld):
                dist = dist - math.sqrt((avgPathX[i + 1] - avgPathX[i]) ** 2 + (avgPathY[i + 1] - avgPathY[i]) ** 2)
        minIndexOld = minIndex
        time = T[j] - T[j - 1]
        vel.append(dist / time)
    vap_mean = np.mean(vel)
    vap_std = np.std(vel)
    return vap_mean, vap_std


def ALH(X, Y, avgPathX, avgPathY):  # promedio del la distancia entre el camino real y el promedio en la trayectoria
    alh = []
    for j in range(len(X)):
        minDist = float('Inf')
        for i in range(len(avgPathX)):
            dist = math.sqrt((X[j] - avgPathX[i]) ** 2 + (Y[j] - avgPathY[i]) ** 2)
            if dist < minDist:
                minDist = dist
        alh.append(minDist)
    alh_mean = np.mean(alh)
    alh_std = np.std(alh)
    return alh_mean, alh_std


def LIN(X, Y, T):
    lin = VSL(X, Y, T) / VCL(X, Y, T)
    return lin


def WOB(X, Y, avgPathX, avgPathY, T):
    vap_mean, vap_std = VAP(X, Y, avgPathX, avgPathY, T)
    wob = vap_mean / VCL(X, Y, T)
    return wob


def STR(X, Y, avgPathX, avgPathY, T):
    vap_mean, vap_std = VAP(X, Y, avgPathX, avgPathY, T)
    stra = VSL(X, Y, T) / vap_mean
    return stra


def BCF(X, Y, avgPathX, avgPathY, T):
    bcf = []
    for j in range(1, len(X)):
        minDist = float('Inf')
        for i in range(len(avgPathX)):
            dist = math.sqrt((X[j] - avgPathX[i]) ** 2 + (Y[j] - avgPathY[i]) ** 2)
            if dist < minDist:
                minIndexNew = i
                minDist = dist
        if j > 1:
            Ax = avgPathX[minIndexOld]
            Ay = avgPathY[minIndexOld]
            Bx = avgPathX[minIndexNew]
            By = avgPathY[minIndexNew]
            discNew = (Bx - Ax) * (Y[j] - Ay) - (By - Ay) * (X[j] - Ax)
            discOld = (Bx - Ax) * (Y[j - 1] - Ay) - (By - Ay) * (X[j - 1] - Ax)
            if discOld * discNew < 0:
                bcf.append(1)
            else:
                bcf.append(0)
        minIndexOld = minIndexNew
    bcf_mean = np.mean(bcf)
    bcf_std = np.std(bcf)
    return bcf_mean, bcf_std


def MAD(X, Y):
    mad = []
    for i in range(1, len(X) - 1):
        if (X[i] - X[i - 1]) != 0:
            pend1 = (Y[i] - Y[i - 1]) / (X[i] - X[i - 1])
            angle1 = math.atan(pend1)
            if (pend1 < 0 and X[i] < X[i - 1]) or (pend1 > 0 and X[i] < X[i - 1]):
                angle1 = angle1 + math.pi
            else:
                angle1 = 2 * math.pi + angle1
        elif Y[i] > Y[i - 1]:
            angle1 = math.pi / 2
        else:
            angle1 = -math.pi / 2
        if (X[i + 1] - X[i]) != 0:
            pend2 = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i])
            angle2 = math.atan(pend2)
            if (pend2 < 0 and X[i + 1] < X[i]) or (pend2 > 0 and X[i + 1] < X[i]):
                angle2 = angle2 + math.pi
            else:
                angle2 = 2 * math.pi + angle2
        elif Y[i + 1] > Y[i]:
            angle2 = math.pi / 2
        else:
            angle2 = -math.pi / 2
        mad.append(abs(angle1 - angle2))
    return np.mean(mad)
