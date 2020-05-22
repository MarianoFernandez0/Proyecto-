from dataset_generator import read_parameters, make_sequence

sequence_parameters, all_population = read_parameters(path='config.txt')
make_sequence(sequence_parameters, all_population)