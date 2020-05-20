def read_file(fileName):

    POS_TRACK_ID = 2
    POS_X = 4
    POS_Y = 5
    POS_T = 7

    X = []
    Y = []
    T = []
    TRACK_ID = []
    with open(fileName,'r') as file:
        lines = file.read().split("\n")
        terminos = lines[1].split(",") #con lines[1] me dirijo a la primera linea de datos
        trackId = int(terminos[POS_TRACK_ID]) #obtengo el primer trackId
        x = []
        y = []
        t = []
        i = 1
        while True:
            #print('hola')
            if lines[i]:
                terminos = lines[i].split(",")
                if trackId != int(terminos[POS_TRACK_ID]): #tengo un nuevo trackId
                    TRACK_ID.append(trackId)
                    trackId = int(terminos[POS_TRACK_ID])
                    #print(trackId)
                    X.append(x)
                    Y.append(y)
                    T.append(t)
                    x = []
                    y = []
                    t = []
                elif t == [] or float(terminos[POS_T]) != t[-1]:
                    x.append(float(terminos[POS_X]))
                    y.append(float(terminos[POS_Y]))
                    t.append(float(terminos[POS_T]))
                    i = i + 1
                else:
                    i = i + 1
            else:
                X.append(x)
                Y.append(y)
                T.append(t)
                TRACK_ID.append(trackId)
                break

    file.close()
    return X,Y,T,TRACK_ID