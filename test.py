import argparse
from data_process import *
from model import *
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def run():
    aid_int_map_activity, aid_int_map_kp = generate_dict("/riseml/workspace/data/child_data_small_new.csv")
    seqs_by_student, num_activities, num_kp = read_file("/riseml/workspace/data/child_data_small_new.csv", aid_int_map_activity, aid_int_map_kp)

    batch_size = 24
    max_seq_len = 201
    config = {"hidden_neurons": [300],
              "embedding_activity_dim": 300,
              "embedding_kp_dim": 300,
              "batch_size": batch_size,
              "keep_prob": 0.6,
              "num_activities": (num_activities - 1) * 4 + 1,
              "num_kp": (num_kp - 1) * 4 + 1,
              "max_seq_len": max_seq_len - 1,
              "learning_rate": 1e-3}

    # random_key('test_key.txt', seqs_by_student)

    _, test_seqs = split_test_dataset('/riseml/workspace/data/test_key_audio.txt', seqs_by_student)
    test_data, test_seq_len = format_data(test_seqs, max_seq_len)

    y_test = test_data[:, :, 2]

    x_test_activities = np.array(test_data)[:, :, 0][:, 1:max_seq_len]
    x_test_kp = np.array(test_data)[:, :, 1][:, 1:max_seq_len]

    x_test_seq_len = np.array(test_seq_len)

    y_test = np.array(y_test)[:, 1:max_seq_len]
    x_pre_score_test = y_test[:, :max_seq_len - 1]

    x_test_activities = tansform_input(x_test_activities, x_pre_score_test, num_activities)
    x_test_kp = tansform_input(x_test_kp, x_pre_score_test, num_kp)

    model = TensorFlowDKT(config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_varibles(sess, filepath="{0}/model_weight_{1}.model".format('/riseml/workspace/data/model', 5))
        for start, end in zip(range(0, len(x_test_activities), batch_size), range(batch_size, len(x_test_activities) + batch_size, batch_size)):
            input_feed = {model.input_activity: x_test_activities[start:end],
                          model.input_kp: x_test_kp[start:end],
                          model.target_id: y_test[start:end],
                          model.sequence_len: x_test_seq_len[start:end],
                          model.keep_prob: 1}
            y_pred = sess.run([model.y_pred, ], input_feed)
            batch_seq_len = x_test_seq_len[start:end]
            batch_y_test = y_test[start:end]
            for num in range(len(y_pred[0])):
                plt.plot(range(batch_seq_len[num]), y_pred[0][num][:batch_seq_len[num]], c='red')
                plt.plot(range(batch_seq_len[num]), batch_y_test[num][:batch_seq_len[num]], c='blue')
                plt.savefig("/riseml/workspace/data/result_test/"+str(start+num)+".png")
                plt.close()



if __name__ == "__main__":
    run()