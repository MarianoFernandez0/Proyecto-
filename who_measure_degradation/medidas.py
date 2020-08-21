import numpy as np
import math
from scipy.spatial.distance import cdist
from itertools import groupby

def avgPath(X, Y, fps):
    NumberNewPoints = 5 if fps < 30 else 7
    xPath = []
    yPath = []

    for i in range(len(X) - 1):
        xvals = np.linspace(X[i], X[i + 1], NumberNewPoints + 2)  # crear 5 puntos en el medio (7 en total)
        yvals = np.linspace(Y[i], Y[i + 1], NumberNewPoints + 2)

        for j in range(len(xvals)):
            # obtengo caminos con muchos puntos intermedios (5 entre cada dos puntos originales)
            xPath.append(xvals[j])
            yPath.append(yvals[j])

    windowSize = NumberNewPoints * 3
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
    xyavgPath = [(avgPathX, avgPathY) for avgPathX, avgPathY in zip(avgPathX, avgPathY)]
    for j in range(1, len(X)):
        # minDist = float('Inf')
        dists = cdist(np.array([[X[j], Y[j]]]), np.array(xyavgPath))
        minIndex = np.argmin(dists)
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
    minIndexOld = 0
    xyavgPath = [(avgPathX, avgPathY) for avgPathX, avgPathY in zip(avgPathX, avgPathY)]

    for j in range(1, len(X)):
        # minDist = float('Inf')
        dists = cdist(np.array([[X[j], Y[j]]]), np.array(xyavgPath))
        minIndexNew = np.argmin(dists)
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
    time_unit = T[1] - T[0]
    freqs = []
    start = -1
    for i in range(1, len(bcf)):
        if bcf[i] < bcf[i-1] and start == -1:
            start = i
        if bcf[i] > bcf[i-1] and start != -1:
            freqs.append(1/((i-start)*time_unit))
            start = -1
    bcf_mean = np.mean(freqs)
    bcf_std = np.std(freqs)
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
