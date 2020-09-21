import cPickle
import numpy as np


track_encodes = cPickle.load(open('../input/dataset_5_15Hz_ennjpdaf/pickles/traj_vec_normal_reverse'))
# print track_encodes

encodes_list = []
for tr in track_encodes:
    encodes_list.append(tr[0][0])
encodes_array = np.array(encodes_list)
print encodes_array
print encodes_array.shape
np.save('../input/dataset_5_15Hz_ennjpdaf/pickles/encodes', encodes_array)
# print encodes_list