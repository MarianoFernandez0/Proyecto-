import os
import pandas
import numpy as np
from sklearn import preprocessing
from argparse import ArgumentParser
from sklearn.cluster import KMeans
import tensorflow as tf
import cv2
import tifffile


def tracks_to_list(tracks):
    tracks_dict = {}
    for i in range(tracks.shape[0]):
        if tracks[i, 0] in tracks_dict.keys():
            tracks_dict[tracks[i, 0]].append([tracks[i, 4], tracks[i, 1], tracks[i, 2]])
        else:
            tracks_dict[tracks[i, 0]] = [[tracks[i, 4], tracks[i, 1], tracks[i, 2]]]

    tracks_list = []
    for k in tracks_dict.keys():
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
                if locCrate<3:
                    rec.append(0)
                    rec.append(0)
                else:
                    rec.append(track_comps[i][2]-track_comps[i-1][2])
                    rec.append(track_comps[i][3]-track_comps[i-1][3])
            track_com_fea.append(rec)
        tracks_feas.append(track_com_fea)
    return tracks_feas


def rolling_window(sample, windowsize=600, offset=300):
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
            pd = pandas.DataFrame(records)
            pdd = pd.describe()

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


def generate_behavior_sequences(data):
    behavior_sequences = []
    for sample in data:
        windows = rolling_window(sample)
        behavior_sequence = behavior_ext(windows)
        # print(len(behavior_sequence))
        behavior_sequences.append(behavior_sequence)
    return behavior_sequences


def generate_normal_behavior_sequence(behavior_sequences):
    # print(np.shape(behavior_sequences))
    behavior_sequences_normal = []
    templist = []
    for item in behavior_sequences:
        for ii in item:
            templist.append(ii)
        # print(len(item))
    # print(len(templist))
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


def trajectory2Vec(input_datas):
    def loopf(prev, i):
        return prev

    # Parameters
    learning_rate = 0.0001
    training_epochs = 300
    display_step = 100

    # Network Parameters
    # the size of the hidden state for the lstm
    # (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
    size = 100
    # 2 different sequences total
    batch_size = 1
    # the maximum steps for both sequences is 5
    max_n_steps = 17
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
            print('Sample:')
            print(j)
            input_len = len(input_data)
            print(input_len)
            defalt = []
            for i in range(0, frame_dim):
                defalt.append(0)
            while len(input_data) < max_n_steps:
                input_data.append(defalt)
            x = np.array(input_data)
            print(np.shape(x[0]))
            x = x.reshape((max_n_steps, batch_size, frame_dim))
            embedding = None
            for epoch in range(training_epochs):
                feed = {seq_input: x, input_length: np.array([input_len])}
                # Fit training using batch data
                _, cost_value, embedding, en_int, de_outs, loss_in = sess.run(
                    [optimizer, loss, enc_state, encoder_inputs, dec_outputs, loss_inputs], feed_dict=feed)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("logits")
                    a = sess.run(y_pred, feed_dict=feed)
                    print("labels")
                    b = sess.run(y_true, feed_dict=feed)

                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
            trajectoryVecs.append(embedding)
            print("Optimization Finished!")
            j = j + 1
        return trajectoryVecs


def get_encodes(trajectories):
    # print 'trajectories:\n', trajectories
    trajectories_list = tracks_to_list(trajectories)
    # print 'trajectories_list:\n', trajectories_list
    trajectories_com = complete_trajectories(trajectories_list)
    # print 'trajectories_com:\n', trajectories_com
    trajectories_feas = compute_feas(trajectories_com)
    # print 'trajectories_feas:\n', trajectories_feas
    behavior_sequences = generate_behavior_sequences(trajectories_feas)
    # print 'behavior_sequences:\n', behavior_sequences
    behavior_sequences_normal = generate_normal_behavior_sequence(behavior_sequences)
    # print 'behavior_sequences_normal:\n', behavior_sequences_normal
    trajectories_vecs = trajectory2Vec(behavior_sequences_normal)
    # print 'trajectories_vecs:\n', trajectories_vecs

    encodes_list = []
    for tr in trajectories_vecs:
        encodes_list.append(tr[0][0])
    encodes_array = np.array(encodes_list)
    print('encodes shape', encodes_array.shape)
    return encodes_array


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tracks', default='../data/real_data/good_video/tracks.csv')
    parser.add_argument('--tif', default='../data/real_data/good_video/video.tif')
    parser.add_argument('--video', default='../data/video_.mp4')
    parser.add_argument('--encodes', default='../data/vector_')
    args = parser.parse_args()

    trajectories = np.genfromtxt(args.tracks, delimiter=',', skip_header=True)  # id,x,y,fluorescence,frame
    # encodes_array = get_encodes(trajectories)
    # np.save(args.encodes, encodes_array)
    #
    # ids = np.unique(trajectories[:, 0])
    # print('ids', len(ids))
    # frames = np.unique(trajectories[:, 4])
    # print('frames', len(frames))
    #
    # km = KMeans(n_clusters=4)
    # km.fit(encodes_array)
    # print(km.labels_)
    #
    # tif = tifffile.TiffFile(args.tif)
    # sequence = tif.asarray()
    # print('sequence ')
    # print(sequence.shape)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(args.video, fourcc, 4, (512, 512))
    #
    # colors = [(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
    # img_old = np.zeros(sequence.shape[1:])
    # for i, frame in enumerate(frames):
    #     frame_tracks = trajectories[trajectories[:, 4] == frame]
    #     # frame_img = cv2.cvtColor(sequence[i, :, :, :], cv2.COLOR_RGB2BGR)
    #     frame_img = sequence[int(frame), :, :, :]
    #     frame_img_2 = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
    #     img = np.zeros(sequence.shape[1:]) + img_old
    #     frame_ids = np.unique(frame_tracks[:, 0])
    #     for i, track_id in enumerate(frame_ids):
    #         track = frame_tracks[frame_tracks[:, 0] == track_id]
    #         img_color = colors[int(km.labels_[np.argwhere(ids == track_id)])]
    #         cv2.circle(img, (int(track[:, 1]), int(track[:, 2])), 2, img_color)
    #     draw_img = frame_img_2 + img
    #     # cv2.imshow(str(frame), draw_img)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     out.write(draw_img.astype(np.uint8))
    #     img_old = img
    # out.release()


    # print 'trajectories:\n', trajectories
    trajectories_list = tracks_to_list(trajectories)
    # print 'trajectories_list:\n', trajectories_list
    trajectories_com = complete_trajectories(trajectories_list)
    # print 'trajectories_com:\n', trajectories_com
    trajectories_feas = compute_feas(trajectories_com)
    # print 'trajectories_feas:\n', trajectories_feas
    behavior_sequences = generate_behavior_sequences(trajectories_feas)
    # print 'behavior_sequences:\n', behavior_sequences
    behavior_sequences_normal = generate_normal_behavior_sequence(behavior_sequences)
    # print 'behavior_sequences_normal:\n', behavior_sequences_normal
    # encoder_input = behavior_sequences_normal
    lstm_size = 100

    # Return states in addition to output
    encoder_input = tf.keras.layers.Input(shape=(None,))
    output, state_h, state_c = tf.keras.layers.LSTM(lstm_size, return_state=True, name="encoder")(encoder_input)
    encoder_state = [state_h, state_c]

    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_input = tf.keras.layers.Input(shape=(None,))
    decoder_output = tf.keras.layers.LSTM(lstm_size, name="decoder")(decoder_input, initial_state=encoder_state)

    model = tf.keras.Model([encoder_input, decoder_input], output)
    model.summary()

