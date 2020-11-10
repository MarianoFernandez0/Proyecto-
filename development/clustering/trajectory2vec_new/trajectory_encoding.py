import os
import cv2
import tifffile
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans


def tracks_to_list(tracks, fps=1, um_per_px=1):
    print 'tracks'
    print tracks.shape
    print len(np.unique(tracks[:, 0]))
    tracks_dict = {}
    for i in range(tracks.shape[0]):
        if tracks[i, 0] in tracks_dict.keys():
            tracks_dict[tracks[i, 0]].append([tracks[i, 4]/fps, tracks[i, 1]*um_per_px, tracks[i, 2]*um_per_px])
        else:
            tracks_dict[tracks[i, 0]] = [[tracks[i, 4]/fps, tracks[i, 1]*um_per_px, tracks[i, 2]*um_per_px]]
    print len(tracks_dict.keys())
    tracks_list = []
    for k in sorted(tracks_dict.keys()):
        # print k
        tracks_list.append(tracks_dict[k])

    return tracks_list


def complete_trajectories(tracks_list):
    # print(tracks_list)
    tracks_comps = []
    for track in tracks_list:
        track_com = []
        for i in range(0, len(track)):
            rec = []
            if i == 0:
                # time, locationC, speedC, rotC
                rec = [0, 0, 0, 0]
            else:
                locC = np.sqrt((track[i][1]-track[i-1][1])**2+(track[i][2]-track[i-1][2])**2)
                rec.append(track[i][0])
                rec.append(locC)
                rec.append(locC/(track[i][0]-track[i-1][0]))
                rec.append(np.arctan((track[i][2]-track[i-1][2]) / (track[i][1]-track[i-1][1])))
            track_com.append(rec)
        tracks_comps.append(track_com)
    return tracks_comps


def compute_feas(tracks_comps):
    tracks_feas = []
    for track_comps in tracks_comps:
        track_com_fea = []
        for i in range(0, len(track_comps)):
            rec = []
            if i == 0:
                # time, locationC, speedC, rotC
                rec = [0, 0, 0, 0]
            else:
                locC = track_comps[i][1]
                locCrate = locC/(track_comps[i][0]-track_comps[i-1][0])
                rec.append(track_comps[i][0])
                rec.append(locCrate)
                # if locCrate<3:
                if locCrate<0.1:
                    rec.append(0)
                    rec.append(0)
                else:
                    rec.append(track_comps[i][2]-track_comps[i-1][2])
                    rec.append(track_comps[i][3]-track_comps[i-1][3])
            track_com_fea.append(rec)
        tracks_feas.append(track_com_fea)
    return tracks_feas


def rolling_window(sample, windowsize=4, offset=2):
    timeLength = sample[len(sample)-1][0]
    windowLength = int(timeLength/offset) + 1
    windows = []
    for i in range(0, windowLength):
        windows.append([])

    for record in sample:
        time = record[0]
        for i in range(0, windowLength):
            if (time > (i*offset)) & (time < (i*offset+windowsize)):
                windows[i].append(record)
    return windows


def behavior_ext(windows):
    behavior_sequence = []
    for window in windows:
        behaviorFeature = []
        records = np.array(window)
        if len(records) != 0:
            df = pd.DataFrame(records)
            pdd = df.describe()

            behaviorFeature.append(pdd[1][1])
            behaviorFeature.append(pdd[2][1])
            behaviorFeature.append(pdd[3][1])

            behaviorFeature.append(pdd[1][3])
            behaviorFeature.append(pdd[2][3])
            behaviorFeature.append(pdd[3][3])

            behaviorFeature.append(pdd[1][4])
            behaviorFeature.append(pdd[2][4])
            behaviorFeature.append(pdd[3][4])

            behaviorFeature.append(pdd[1][5])
            behaviorFeature.append(pdd[2][5])
            behaviorFeature.append(pdd[3][5])

            behaviorFeature.append(pdd[1][6])
            behaviorFeature.append(pdd[2][6])
            behaviorFeature.append(pdd[3][6])

            behaviorFeature.append(pdd[1][7])
            behaviorFeature.append(pdd[2][7])
            behaviorFeature.append(pdd[3][7])

            behavior_sequence.append(behaviorFeature)
    return behavior_sequence


def generate_behavior_sequences(data, windowsize=4, offset=2):
    behavior_sequences = []
    for i, sample in enumerate(data):
        windows = rolling_window(sample, windowsize=windowsize, offset=offset)
        behavior_sequence = behavior_ext(windows)
        behavior_sequences.append(behavior_sequence)
    return behavior_sequences


def     generate_normal_behavior_sequence(behavior_sequences):
    behavior_sequences_normal = []
    templist = []
    for item in behavior_sequences:
        for ii in item:
            templist.append(ii)
    min_max_scaler = preprocessing.MinMaxScaler()

    templist_normal = min_max_scaler.fit_transform(templist).tolist()
    index = 0
    for item in behavior_sequences:
        behavior_sequence_normal = []
        for ii in item:
            behavior_sequence_normal.append(templist_normal[index])
            index = index + 1
        # print(len(behavior_sequence_normal))
        behavior_sequences_normal.append(behavior_sequence_normal)
    # print(index)
    # print(np.shape(behavior_sequences_normal))
    return behavior_sequences_normal


def trajectory2Vec(input_datas, size=100):
    def loopf(prev, i):
        return prev

    # Parameters
    learning_rate = 0.0001
    training_epochs = 300
    display_step = 100

    # Network Parameters
    # the size of the hidden state for the lstm
    # (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
    # size = 100  # ORIGINAL
    size = size
    # 2 different sequences total
    batch_size = 1
    # the maximum steps for both sequences is 5
    max_n_steps = 60
    # each element/frame of the sequence has dimension of 3
    frame_dim = 18

    input_length = tf.placeholder(tf.int32)

    initializer = tf.random_uniform_initializer(-1, 1)

    # the sequences, has n steps of maximum size
    # seq_input = tf.placeholder(tf.float32, [batch_size, max_n_steps, frame_dim])
    seq_input = tf.placeholder(tf.float32, [max_n_steps, batch_size, frame_dim])
    # what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]

    # inputs for rnn needs to be a list, each item/frame being a timestep.
    # we need to split our input into each timestep, and reshape it because split keeps dims by default

    useful_input = seq_input[0:input_length[0]]
    loss_inputs = [tf.reshape(useful_input, [-1])]
    encoder_inputs = [item for item in tf.unpack(seq_input)]
    # if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
    # and target size equal encoder size plus 1. For simplicity, here I droped the last one.
    decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
    targets = encoder_inputs

    # basic LSTM seq2seq model
    cell = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, use_peepholes=True)
    _, enc_state = tf.nn.rnn(cell, encoder_inputs, sequence_length=input_length[0], dtype=tf.float32)
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, frame_dim)
    dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loopf)

    # flatten the prediction and target to compute squared error loss
    y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
    y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

    # Define loss and optimizer, minimize the squared error
    loss = 0
    for i in range(len(loss_inputs)):
        loss += tf.reduce_sum(tf.square(tf.sub(y_pred[i], y_true[len(loss_inputs) - i - 1])))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        trajectoryVecs = []
        j = 0
        for input_data in input_datas:
            print 'Sample:'
            print j
            input_len = len(input_data)
            print input_len
            defalt = []
            for i in range(0, frame_dim):
                defalt.append(0)
            while len(input_data) < max_n_steps:
                input_data.append(defalt)
            x = np.array(input_data)
            print np.shape(x[0])
            print x.shape
            x = x.reshape((max_n_steps, batch_size, frame_dim))
            embedding = None
            for epoch in range(training_epochs):
                feed = {seq_input: x, input_length: np.array([input_len])}
                # Fit training using batch data
                _, cost_value, embedding, en_int, de_outs, loss_in = sess.run(
                    [optimizer, loss, enc_state, encoder_inputs, dec_outputs, loss_inputs], feed_dict=feed)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "logits"
                    a = sess.run(y_pred, feed_dict=feed)
                    print "labels"
                    b = sess.run(y_true, feed_dict=feed)

                    print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value)
            trajectoryVecs.append(embedding)
            print "Optimization Finished!"
            j = j + 1
        return trajectoryVecs


def get_encodes(trajectories):
    print 'trajectories:\n', len(trajectories), trajectories[0, :]
    trajectories_list = tracks_to_list(trajectories)
    print 'trajectories_list:\n', len(trajectories_list), trajectories_list[0][:3]
    trajectories_com = complete_trajectories(trajectories_list)
    print 'trajectories_com:\n', len(trajectories_com), trajectories_com[0][:3]
    trajectories_feas = compute_feas(trajectories_com)
    print 'trajectories_feas:\n', len(trajectories_feas), trajectories_feas[0][:3]
    behavior_sequences = generate_behavior_sequences(trajectories_feas, windowsize=10, offset=5)
    print 'behavior_sequences:\n', len(behavior_sequences), behavior_sequences[0][:3]
    behavior_sequences_normal = generate_normal_behavior_sequence(behavior_sequences)
    print 'behavior_sequences_normal:\n', len(behavior_sequences_normal), behavior_sequences_normal[0][:3]
    trajectories_vecs = trajectory2Vec(behavior_sequences_normal, size=20)
    print 'trajectories_vecs:\n', len(trajectories_vecs), trajectories_vecs[0]

    encodes_list = []
    for tr in trajectories_vecs:
        encodes_list.append(tr[0][0])
    encodes_array = np.array(encodes_list)
    print 'encodes shape', encodes_array.shape
    return encodes_array


def draw_labels(sequence, trajectories, labels, data_dir):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(os.path.join(data_dir, 'video.mp4'), fourcc, 4, (512, 512))

    ids = np.unique(trajectories[:, 0])
    frames = np.unique(trajectories[:, 4])
    colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]

    img_old = np.zeros((sequence.shape[1], sequence.shape[2], 3))
    for i, frame in enumerate(frames):
        frame_tracks = trajectories[trajectories[:, 4] == frame]
        if len(sequence.shape) < 4:
            frame_img = cv2.cvtColor(sequence[int(frame), :, :], cv2.COLOR_GRAY2BGR)
        else:
            frame_img = sequence[int(frame), :, :, :]
        img = np.zeros((sequence.shape[1], sequence.shape[2], 3)) + img_old
        frame_ids = np.unique(frame_tracks[:, 0])
        for k, track_id in enumerate(frame_ids):
            track = frame_tracks[frame_tracks[:, 0] == track_id]
            img_color = colors[int(labels[np.argwhere(ids == track_id)])]
            cv2.circle(img, (int(track[:, 1]), int(track[:, 2])), 2, img_color)
            cv2.putText(frame_img, str(int(track_id)), (int(track[:, 1]), int(track[:, 2])), 2, 0.5, color=img_color)
        draw_img = frame_img.copy()
        draw_img[img != 0] = img[img != 0]
        out_vid.write(draw_img.astype(np.uint8))
        img_old = img
    out_vid.release()


def plot_fluo(tracks, tracks_info, folder):
    mean_fluorescence = []
    for k, track_id in enumerate(ids):
        mean_fluorescence.append(np.nanmean(tracks[tracks[:, 0] == track_id][:, 3]))
    tracks_info['mean_fluorescence'] = mean_fluorescence

    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    plt.scatter(tracks_info['label'], tracks_info['mean_fluorescence'], alpha=0.5)
    plt.xticks([0, 1, 2, 3])
    plt.title('Fluorescence distribution')
    plt.xlabel('label')
    plt.ylabel('mean_fluorescence')
    ax.yaxis.grid(True)
    # plt.show()
    fig.savefig(os.path.join(folder, 'fluorescence_distribution.png'), dpi=300)

    labels_groups = tracks_info.groupby('label')['mean_fluorescence'].describe()
    # print(labels_groups)
    width = 0.3
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    # ax.bar(labels_groups.index + width*(-2), labels_groups['min'], width, label='min')
    ax.bar(labels_groups.index + width * (0), labels_groups['mean'], width, label='mean, err=std', alpha=0.7)
    ax.errorbar(labels_groups.index, labels_groups['mean'], yerr=labels_groups['std'], label='std',
                fmt=' ', ecolor='k', capsize=10)
    # ax.bar(labels_groups.index + width*(0), labels_groups['50%'], width, label='median')
    # ax.bar(labels_groups.index + width*(1), labels_groups['max'], width, label='max')
    ax.yaxis.grid(True)
    plt.xticks([0, 1, 2, 3])
    plt.legend()
    plt.title('Mean Fluorescence')
    ax.set_xlabel('label')
    ax.set_ylabel('mean_fluorescence')
    # plt.show()
    fig.savefig(os.path.join(folder, 'mean_fluorescence.png'), dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/individual/1_026')  # id,x,y,fluorescence,frame
    parser.add_argument('--gen_encodes', action='store_true')
    parser.add_argument('--fluorescence', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir
    trajectories = np.genfromtxt(os.path.join(data_dir, 'trajectories.csv'), delimiter=',', skip_header=True)

    if args.gen_encodes:
        encodes_array = get_encodes(trajectories)
        np.save(os.path.join(data_dir, 'encodes'), encodes_array)
    else:
        encodes_array = np.load(os.path.join(data_dir, 'encodes.npy'))

    km = KMeans(n_clusters=4)
    km.fit(encodes_array)

    sequence = tifffile.TiffFile(os.path.join(data_dir, 'video.tif')).asarray()
    draw_labels(sequence, trajectories, km.labels_, data_dir)

    ids = np.unique(trajectories[:, 0])
    tracks_labels = pd.DataFrame(columns=['id', 'label'])
    tracks_labels['id'] = ids.astype(int)
    tracks_labels['label'] = km.labels_
    tracks_labels.to_csv(os.path.join(data_dir, 'id_labels.csv'), index=False)
    print tracks_labels.groupby('label').describe()['id']['count'].astype(int)

    if args.fluorescence:
        plot_fluo(trajectories, tracks_labels, data_dir)
