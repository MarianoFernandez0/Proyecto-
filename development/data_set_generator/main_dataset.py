from dataset_generator import read_parameters, make_sequence
import sys

def __init__():
    """
    Function that reads the path in for the configuration file,
    and then make the sequences based on the configuration file.
    The call must be as follow:
    main_dataset.py path/to/config/file.txt
    """
    path = sys.argv[1]
    if not path:
        path = 'config.txt'
    sequence_parameters, all_population = read_parameters(path=path)
    make_sequence(sequence_parameters, all_population)
    return

__init__()