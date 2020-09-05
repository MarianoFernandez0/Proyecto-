import numpy as np
import configparser


class Spermatozoa:
    def __init__(self, width, height, angle, position):
        self.widht = width
        self.height = height
        self.angle = angle
        self.position = position


class Track:
    next_id = 0
    def __init__(self, characteristics, first_frame, final_frame, time_step, frame_shape):
        self.track_id = Track.next_id + 1
        self.type = characteristics['type']
        self.vap = characteristics['vap']
        self.Tp = characteristics['Tp']
        self.bcf = characteristics['bcf']
        self.alh = characteristics['alh']
        self.positions = {first_frame:
                              {'pos': np.array([np.random.uniform(frame_shape[0]), np.random.uniform(frame_shape[1])]),
                               'angle': np.random.uniform(0, 2 * np.pi)}}
        self.first_frame = first_frame
        self.final_frame = final_frame
        self.time_step = time_step
        self.frame_shape = frame_shape
        self.fill_positions()

    def check_inside_frame(self, pos):
        return np.logical_and((0 <= pos[0] <= self.frame_shape[0]), (0 <= pos[1] <= self.frame_shape[1]))

    def fill_positions(self):
        for frame in range(self.first_frame + 1, self.final_frame):
            prev_pos = self.positions[frame - 1]['pos']
            next_angle = np.random.normal(0, 1) * np.sqrt(2 / self.Tp) * self.time_step
            next_position = prev_pos + self.vap * np.array([np.cos(next_angle), np.sin(next_angle)]) * self.time_step
            if not self.check_inside_frame(next_position):
                break
            self.positions[frame] = {'pos': next_position, 'angle': next_angle}


class ConfigData:
    def __init__(self, path_config_file):
        config = configparser.ConfigParser()
        config.read(path_config_file)
        config.sections()
        print("Reading configuration file...")

        self.path_data_out = config["Output path"]["PATH_OUTPUT_DATA_CSV"]
        self.path_seq_out = config["Output path"]["PATH_OUTPUT_DATA_SEQ"]
        self.frame_shape = config["Sequence parameters"]["height_width_duration"].split()[:-1]
        self.duration = config["Sequence parameters"]["height_width_duration"].split()[-1]
        self.rgb = (config["Sequence parameters"]["rgb"]).lower() == "true"
        self.std_blur = float(config["Sequence parameters"]["std_blur"])
        self.noise_type = config["Sequence parameters"]["noise_type"]
        self.noise_params = np.array(config["Sequence parameters"]["noise_params"].split(),
                                     dtype=np.float)
        self.low_limit = float(config["Sequence parameters"]["low_limit"])
        self.file_format = config["Sequence parameters"]["file_format"].split()
        self.file_name = config["Sequence parameters"]["file_name"]
        self.frame_rate = np.array(config["Sequence parameters"]["frame_rate"].split(),
                                   dtype=np.float)
        self.seed = int(config["Sequence parameters"]["seed"])
        self.resolution = float(config["Sequence parameters"]["resolution"])

        # load populations
        self.populations = []
        for pop in ((config.sections())[2:]):
            population = Population(int(config[pop]["tot_particles"]),
                                    (config[pop]["movement_type"]).lower(),
                                    np.array(config[pop]["color"].split(), dtype=np.float),
                                    np.array(config[pop]["mean"].split(), dtype=np.float),
                                    (np.array(config[pop]["cov_mean"].split(), dtype=np.float)).reshape(2, 2),
                                    float(config[pop]["VAP"]),
                                    float(config[pop]["VAP_deviation"]),
                                    float(config[pop]["Tp"]),
                                    (config[pop]["head_displ"]).lower() == "true",
                                    float(config[pop]["std_depth"]),
                                    float(config[pop]["ALH_mean"]),
                                    float(config[pop]["ALH_std"]),
                                    float(config[pop]["BCF_mean"]),
                                    float(config[pop]["BCF_std"]))
            self.populations.append(population)
            self.sequence = {}

    def make_sequence(self):
        fps_base = np.lcm(self.frame_rate)
        for population in self.populations:
            tracks = [Track()]


class Population:
    def __init__(self, particles, mov_type, color, mean, cov_mean, vap, vap_deviation, Tp, head_displ, std_depth,
                 ALH_mean, ALH_std, BCP_mean, BCP_std):
        self.particles = particles
        self.mov_type = mov_type
        self.color = color
        self.mean = mean
        self.cov_mean = cov_mean
        self.vap = vap
        self.vap_deviation = vap_deviation
        self.Tp = Tp
        self.head_displ = head_displ
        self.std_depth = std_depth
        self.ALH_mean = ALH_mean
        self.ALH_std = ALH_std
        self.BCP_mean = BCP_mean
        self.BCP_std = BCP_std