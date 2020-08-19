from dataset_generator_new import read_parameters, make_sequence
import sys

if __name__ =="__main__":

    path = 'configs/dataset_2(40Hz).txt'
    if len(sys.argv) > 1:
        path = sys.argv[1]
    sequence_parameters, all_population = read_parameters(path=path)
    make_sequence(sequence_parameters, all_population)
