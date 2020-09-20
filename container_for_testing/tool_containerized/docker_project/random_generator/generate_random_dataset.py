import os
import numpy as np
from random_generator.dataset_generator_new import read_parameters, make_sequence

def generate_config_file(path_out='random_generator/auto_config'):
    tot_particles = np.random.uniform(150, 300)
    proportions = np.random.dirichlet(np.ones(4), size=1)
    particles = np.round(tot_particles*proportions)[0]
    name = 'config_file.txt'
    os.makedirs(path_out, exist_ok=True)
    with open('random_generator/base.txt', 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(range(22, 23+22*3, 19)):
        lines[line] = lines[line].replace(':', ': {}'.format(int(particles[i])))
    with open(os.path.join(path_out, name), 'w') as f:
        f.writelines(lines)

    sequence_parameters, all_population = read_parameters('random_generator/auto_config/config_file.txt')
    make_sequence(sequence_parameters, all_population)

if __name__ == '__main__':
    generate_config_file()
