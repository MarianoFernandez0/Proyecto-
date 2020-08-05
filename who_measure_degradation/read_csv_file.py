def read_csv_file(df):
    
    TRACK_ID = []
    X = []
    Y = []
    T = []

    for i in range(df['id'].max()+1):
        TRACK_ID.append(i)
    
        new_out = df.loc[df['id']==i]
        X.append(new_out['x'].to_list())
        Y.append(new_out['y'].to_list())
        T.append(new_out['frame'].to_list())
    
    return X,Y,T,TRACK_ID 