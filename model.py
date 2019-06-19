import tensorflow as tf


class Network:
    def __init__(self, voc_size):
        self._get_params(voc_size)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_input_part()
            self._build_emb_part()
            self._Prenet()
            self._ConvLayer()
            self.enc = self.prenet + self.conv1d_pro  # Residual connection
            self._Highway_layers()
            self._BiRNN()
            self._build_other_part()
            self._get_conf()

    def _get_params(self, voc_size):
        self.voc_size = voc_size
        self.t_size = 50
        self.emb_size = 300
        self.unit_fc = [self.emb_size, self.emb_size // 2]
        self.drop_rate = 0.5
        self.K = 16
        self.n_highway_block = 4
        self.lr = 1e-3

    def _build_input_part(self):
        '''
        构建输入部分
        :return:
        '''
        self.X = tf.placeholder(tf.int32, [None, self.t_size])
        self.Y = tf.placeholder(tf.int32, [None, self.t_size])
        self.global_step = tf.Variable(tf.zeros([], tf.int32),
                                       name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool)  # 训练标识位

    def _build_emb_part(self):
        '''
        构建embedding部分
        :return:
        '''
        with tf.variable_scope('spell_emb'):
            lookup_table = tf.get_variable(dtype=tf.float32, shape=[self.voc_size, self.emb_size],
                                           initializer=tf.truncated_normal_initializer(mean=0,
                                                                                       stddev=0.01),
                                           name='emb_lookup')
            lookup_table = tf.concat((tf.zeros([1, self.emb_size]),
                                      lookup_table[1:, :]), axis=0)  # Empty对应的idx为0，将其emb全设为0
            self.spell_emb = tf.nn.embedding_lookup(lookup_table, self.X)  # (None,t_size,emb_size)

    def _Prenet(self):
        '''
        Pre-net
        :return:
        '''
        with tf.variable_scope('Pre-net'):
            prenet = tf.layers.dense(self.spell_emb, units=self.unit_fc[0],
                                     activation=tf.nn.relu)  # (None,t_size,unit_fc[0])
            prenet = tf.layers.dropout(prenet, rate=self.drop_rate,
                                       training=self.is_training)  # (None,t_size,unit_fc[0])
            prenet = tf.layers.dense(prenet, units=self.unit_fc[1],
                                     activation=tf.nn.relu)  # (None,t_size,unit_fc[1])
            self.prenet = tf.layers.dropout(prenet, rate=self.drop_rate,
                                            training=self.is_training)  # (None,t_size,unit_fc[1])

    def _ConvLayer(self):
        '''
        Conv1D bank + stacking + Conv1D projections
        :return:
        '''
        with tf.variable_scope('Conv1D_bank'):
            n_filters = self.emb_size // 2

            # 使用[1,K]个大小的卷积核提取信息，并拼接在一起，同TextCNN
            # k=1
            conv1d_bank = tf.layers.conv1d(self.prenet, filters=n_filters, kernel_size=1,
                                           padding='same', use_bias=False)  # (None,t_size,n_filters)

            # k=2,3,...,K
            for k in range(2, self.K + 1):
                conv = tf.layers.conv1d(self.prenet, filters=n_filters, kernel_size=k,
                                        padding='same', use_bias=False)
                conv1d_bank = tf.concat((conv1d_bank, conv),
                                        axis=-1)  # (None,t_size,k*n_filters)

            conv1d_bank = fused_batch_norm(conv1d_bank, training=self.is_training,
                                           act_f=tf.nn.relu)  # (None,t_size,K*n_filters)

        # 在t维度上做maxpool，同TextCNN
        max_pooling = tf.layers.max_pooling1d(conv1d_bank, pool_size=2, strides=1,
                                              padding='same')  # (None,t_size,K*n_filters)

        with tf.variable_scope('Conv1d_projections'):
            conv1d_pro = tf.layers.conv1d(max_pooling, filters=n_filters, kernel_size=5,
                                          padding='same', use_bias=False)  # (None,t_size,n_filters)
            conv1d_pro = fused_batch_norm(conv1d_pro, training=self.is_training, act_f=tf.nn.relu)

            conv1d_pro = tf.layers.conv1d(conv1d_pro, filters=n_filters, kernel_size=5,
                                          padding='same', use_bias=False)  # (None,t_size,n_filters)
            self.conv1d_pro = fused_batch_norm(conv1d_pro, training=self.is_training, act_f=tf.nn.relu)

    def _Highway_layers(self):
        '''
        Highway layers
        :return:
        '''
        with tf.name_scope('Highway_layers'):
            for i in range(self.n_highway_block):
                self.enc = highway_block(self.enc, units=self.emb_size // 2,
                                         scope='highway_{}'.format(i))  # (None, t_size, emb_size//2)

    def _BiRNN(self):
        '''
        Bidirectional RNN
        :return:
        '''
        gru_fw = tf.nn.rnn_cell.GRUCell(self.emb_size // 2)
        gru_bw = tf.nn.rnn_cell.GRUCell(self.emb_size // 2)
        rnn_out, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw, gru_bw, self.enc,
                                                     dtype=tf.float32)

        # (None,None,emb_size//2*2)，双向RNN*2
        self.enc = tf.concat(rnn_out, axis=2)

    def _build_other_part(self):
        '''
        构建剩余的网络部分
        :return:
        '''
        logits = tf.layers.dense(self.enc, self.voc_size, use_bias=False, name='logit')
        self.preds = tf.to_int32(tf.arg_max(logits, dimension=-1))

        with tf.name_scope('Eval'):
            non_empty_mask = tf.to_float(tf.not_equal(self.Y,
                                                      tf.zeros_like(self.Y)))  # 0代表Empty，不参与计算
            all_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y,
                                                                      logits=logits)  # 无差别loss
            self.loss = tf.reduce_sum(all_loss * non_empty_mask) / \
                        tf.reduce_sum(non_empty_mask)  # 非空loss
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.Y)) * non_empty_mask) / \
                       tf.reduce_sum(non_empty_mask)

        # train_op
        with tf.name_scope('train_op'):
            self.glob_step = tf.Variable(0, name='global_step', trainable=False)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr) \
                    .minimize(self.loss, global_step=self.glob_step)

    def _get_conf(self):
        self.init = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # 按需使用显存


def fused_batch_norm(inputs, training=True, act_f=None):
    '''
    fused batch norm
    :param inputs:
    :param training: training flag
    :param act_f: activation function
    :return:
    '''
    # 把维度扩展成4维，然后使用更快的fusedBN
    shape_I = inputs.get_shape()
    rank_I = shape_I.ndims

    if rank_I in [2, 3, 4]:
        if rank_I == 2:
            inputs = tf.expand_dims(inputs, axis=1)  # (X, X, 1)
            inputs = tf.expand_dims(inputs, axis=2)  # (X, X, 1, 1)
        elif rank_I == 3:
            inputs = tf.expand_dims(inputs, axis=1)  # (X, X, X, 1)

    inputs = tf.layers.batch_normalization(inputs, training=training,
                                           fused=True)

    # 恢复成原来的维度
    if rank_I == 2:
        inputs = tf.squeeze(inputs, axis=[1, 2])  # (X, X)
    elif rank_I == 3:
        inputs = tf.squeeze(inputs, axis=1)  # (X, X, X)

    if act_f:
        inputs = act_f(inputs)

    return inputs


def highway_block(inputs, units, scope=None):
    '''
    highway network block
    :param inputs:
    :param units: FC units
    :param scope:
    :return:
    '''
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs, units, activation=tf.nn.relu, name='H')
        T = tf.layers.dense(inputs, units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name='T')

    return H * T + inputs * (1 - T)
