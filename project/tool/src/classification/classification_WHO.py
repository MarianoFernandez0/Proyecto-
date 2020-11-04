
def classification(df_medidas_who):
    motility_type = []

    for index, row in df_medidas_who.iterrows():

        motility = str('')
        vsl = row['vsl']
        lin = row['lin']

        if vsl > 25.0 and lin > 0.58:
            motility = str('is_straightline_progressive')

        if vsl < 25.0 and vsl > 10.0 and lin > 0.58:
            motility = str('is_straight_slow_progressive')

        if vsl > 10.0 and lin < 0.58:
            motility = str('is_nonstraight_progressive')

        if vsl < 10.0:
            motility = str('is_non_progressive')

        motility_type.append(motility)

    df_medidas_who.insert(len(df_medidas_who.columns), 'motility_type', motility_type, True)
    return df_medidas_who


if __name__ == '__main__':
    import pandas as pd

    medidas_who = pd.read_csv("prueba.csv")
    medidas_who = classification(medidas_who)
    medidas_who.to_csv("medidas_who+classification.csv")
