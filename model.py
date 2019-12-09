import tensorflow as tf


class TensorFlowDKT(object):
    def __init__(self, config):
        self.hidden_neurons = hidden_neurons = config["hidden_neurons"]
        self.num_activities = config["num_activities"]
        self.num_kp = config["num_kp"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.embedding_dim_activity = config["embedding_activity_dim"]
        self.embedding_dim_kp = config["embedding_kp_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.keep_prob_value = config["keep_prob"]

        self.input_activity = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.input_kp = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.sequence_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob
        self.target_id = tf.placeholder(tf.float32, [None, self.max_seq_len])

        init_random_activity = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
        init_random_kp = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

        embedding_activity = tf.get_variable('embedding_activity', [self.num_activities, self.embedding_dim_activity],
                                           initializer=init_random_activity)
        embedding_kp = tf.get_variable('embedding_kp', [self.num_kp, self.embedding_dim_kp],
                                             initializer=init_random_kp)

        embedding_activity_inputs = tf.nn.embedding_lookup(embedding_activity, self.input_activity)
        embedding_kp_inputs = tf.nn.embedding_lookup(embedding_kp, self.input_kp)
        embedding_inputs = tf.concat([embedding_activity_inputs, embedding_kp_inputs], -1)
        # create rnn cell
        hidden_layers = []
        for idx, hidden_size in enumerate(hidden_neurons):
            lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # dynamic rnn
        state_series, self.current_state = tf.nn.dynamic_rnn(cell = self.hidden_cell,
                                                             inputs = embedding_inputs,
                                                             sequence_length = self.sequence_len,
                                                             dtype = tf.float32)

        # output layer
        self.state_series = tf.reshape(state_series, [-1, hidden_neurons[-1]])
        self.logits = tf.layers.dense(self.state_series, 1, activation = tf.sigmoid, name = 'out_put')
        self.y_pred = tf.reshape(self.logits, [-1, self.max_seq_len])

        # compute loss
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_pred - self.target_id), -1), -1)

        self.optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def save_varibles(self, sess, filepath):
        saver = tf.train.Saver()
        saver.save(sess, filepath)

    def load_varibles(self, sess, filepath):
        saver = tf.train.Saver()
        saver.restore(sess, filepath)
