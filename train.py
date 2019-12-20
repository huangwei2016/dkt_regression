from data_process import *
from model import *
import tensorflow as tf
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def run():
    aid_int_map_activity, aid_int_map_kp = generate_dict("/workspace/child_event_gpu_audio.csv")
    print len(aid_int_map_activity)
    seqs_by_student, num_activities, num_kp = read_file("/workspace/child_event_gpu_audio.csv", aid_int_map_activity, aid_int_map_kp)
    random_key('/workspace/test_key_audio.txt', seqs_by_student)
    batch_size = 50
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

    train_seqs, _ = split_test_dataset('/workspace/test_key_audio.txt', seqs_by_student)

    data, seq_len = format_data(train_seqs, max_seq_len)
    x_train, y_train, x_validate, y_validate, = split_validate_dataset(data,seq_len)
    x_train_record, x_train_seq_len = zip(*x_train)
    x_validate_record, x_validate_seq_len = zip(*x_validate)

    x_train_activities = np.array(x_train_record)[:, :, 0][:, 1:max_seq_len]
    x_validate_activities = np.array(x_validate_record)[:, :, 0][:, 1:max_seq_len]
    x_train_kp = np.array(x_train_record)[:, :, 1][:, 1:max_seq_len]
    x_validate_kp = np.array(x_validate_record)[:, :, 1][:, 1:max_seq_len]

    x_train_seq_len = np.array(x_train_seq_len)
    x_validate_seq_len = np.array(x_validate_seq_len)

    y_train_target = np.array(y_train)[:, 1:max_seq_len]
    x_pre_score_train = np.array(y_train)[:, :max_seq_len - 1]

    y_validate_target = np.array(y_validate)[:, 1:max_seq_len]
    x_pre_score_validate = np.array(y_validate)[:, :max_seq_len - 1]


    x_train_activities = tansform_input(x_train_activities, x_pre_score_train, num_activities)
    x_train_kp = tansform_input(x_train_kp, x_pre_score_train, num_kp)
    x_validate_activities = tansform_input(x_validate_activities, x_pre_score_validate, num_activities)
    x_validate_kp = tansform_input(x_validate_kp, x_pre_score_validate, num_kp)

    model = TensorFlowDKT(config)

    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True

    with tf.Session(config=config_gpu) as sess:
        sess.run(tf.global_variables_initializer())
        loss_min = sys.float_info.max
        for epoch in range(50):
            for start, end in zip(range(0, len(x_train_activities), batch_size), range(batch_size, len(x_train_activities) + batch_size, batch_size)):
                input_feed = {model.input_activity: x_train_activities[start:end],
                              model.input_kp: x_train_kp[start:end],
                              model.target_id: y_train_target[start:end],
                              model.sequence_len: x_train_seq_len[start:end],
                              model.keep_prob: config['keep_prob']}
                loss,_ = sess.run([model.loss, model.optim], input_feed)
            loss_total = 0.
            loss_count = 0
            for start, end in zip(range(0, len(x_validate_activities), batch_size), range(batch_size, len(x_validate_activities) + batch_size, batch_size)):
                input_feed = {model.input_activity: x_validate_activities[start:end],
                              model.input_kp: x_validate_kp[start:end],
                              model.target_id: y_validate_target[start:end],
                              model.sequence_len: x_validate_seq_len[start:end],
                              model.keep_prob: 1}
                loss = sess.run([model.loss], input_feed)
                loss_total += loss[0]
                loss_count += 1
            print "epoch:" + str(epoch)
            print float(loss_total / loss_count)
            if loss_min > loss_total:
                loss_min = loss_total
                model.save_varibles(sess, filepath="{0}/model_weight_{1}.model".format('/workspace/model/', epoch))

if __name__ == "__main__":
    run()

