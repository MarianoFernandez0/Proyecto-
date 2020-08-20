def read_csv_file(df):
    TRACK_ID = []
    X = []
    Y = []
    F = []
    fluo = []

    for i in range(df['id'].max() + 1):
        TRACK_ID.append(i)
        new_out = df.loc[df['id'] == i]
        X.append(new_out['x'].to_list())
        Y.append(new_out['y'].to_list())
        F.append(new_out['frame'].to_list())
        fluo.append(new_out['fluorescence'].to_numpy())

    return X, Y, F, TRACK_ID, fluo
