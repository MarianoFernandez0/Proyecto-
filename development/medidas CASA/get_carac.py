import numpy as np
import math
from medidasCASA import avgPath,VCL,VSL,VAP,ALH,LIN,WOB,STR,BCF,MAD

def get_carac(X,Y,T,min_detections):

    allX = X
    allY = Y
    allT = T
    
    CARAC_WHO = []   
    
    for i in range(len(allX)):
        
        X = allX[i]
        Y = allY[i]
        T = allT[i]
        
        if ((np.shape(X)[0]) > min_detections):  
            avgPathX, avgPathY = avgPath(X,Y)
            vcl = VCL(X,Y,T)
            vsl = VSL(X,Y,T)
            vap = VAP(X,Y,avgPathX,avgPathY,T)
            alh = ALH(X,Y,avgPathX,avgPathY)
            lin = LIN(X,Y,T)
            wob = WOB(X,Y,avgPathX,avgPathY,T)
            stra = STR(X,Y,avgPathX,avgPathY,T)
            bcf = BCF(X,Y,avgPathX,avgPathY,T)
            mad = MAD(X,Y)
            carac_who = [vcl,vsl,vap,alh,lin,wob,stra,bcf,mad]
            
        else:
            carac_who = [0,0,0,0,0,0,0,0,0]
        
        
        CARAC_WHO.append(carac_who)


    return CARAC_WHO