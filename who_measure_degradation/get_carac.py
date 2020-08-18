import numpy as np
from medidas import VSL, VCL, VAP, ALH, LIN, WOB, STR, BCF, MAD, avgPath


def get_carac(TRACK_ID, X, Y, F, fps, fluo, min_detections=3):
    allTRACK_ID = TRACK_ID
    allX = X
    allY = Y
    CARAC_WHO = []

    for i in range(len(allX)):

        TRACK_ID = allTRACK_ID[i]
        X = allX[i]
        Y = allY[i]
        T = [float(f) / fps for f in F[i]]

        if (np.shape(X)[0]) > min_detections:
            avgPathX, avgPathY = avgPath(X, Y, fps)
            vcl = VCL(X, Y, T)
            vsl = VSL(X, Y, T)
            vap_mean, vap_std = VAP(X, Y, avgPathX, avgPathY, T)
            alh_mean, alh_std = ALH(X, Y, avgPathX, avgPathY)
            lin = LIN(X, Y, T)
            wob = WOB(X, Y, avgPathX, avgPathY, T)
            stra = STR(X, Y, avgPathX, avgPathY, T)
            bcf_mean, bcf_std = BCF(X, Y, avgPathX, avgPathY, T)
            mad = MAD(X, Y)
            carac_who = [TRACK_ID, vcl, vsl, vap_mean, vap_std, alh_mean, alh_std, lin, wob, stra, bcf_mean, bcf_std,
                         mad, np.max(fluo[i])]

        else:
            carac_who = [TRACK_ID, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        CARAC_WHO.append(carac_who)

    return CARAC_WHO
