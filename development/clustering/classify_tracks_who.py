import os
import pandas as pd
from argparse import ArgumentParser
from project.tool.src.classification.classification_WHO import classification
from project.tool.src.who_measures.get_who_measures import get_casa_measures

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/simulation_30hz')
    args = parser.parse_args()

    tracks_file = os.path.join(args.data_dir, 'trajectories.csv')
    who_file = os.path.join(args.data_dir, 'who_measures.csv')
    get_casa_measures(tracks_file, who_file, 1, 30)

    classification_file = os.path.join(args.data_dir, 'who_classification.csv')
    df_measures = pd.read_csv(who_file)
    df_classified = classification(df_measures)
    df_classified.to_csv(classification_file)