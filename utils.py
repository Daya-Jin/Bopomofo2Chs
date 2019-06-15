import tensorflow as tf

def batch_norm(inputs, training=True, act_f=None):
    '''
    自定义batch_norm，使用fusedBN
    '''
    # 把维度扩展成4维，然后使用更快的fusedBN
    shape_I = inputs.get_shape()
    rank_I = shape_I.ndims

    if rank_I in [2, 3, 4]:
        if rank_I == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif rank_I == 3:
            inputs = tf.expand_dims(inputs, axis=1)

    inputs = tf.layers.batch_normalization(inputs,
                                           training=training, fused=True)

    # 恢复成原来的维度
    if rank_I == 2:
        inputs = tf.squeeze(inputs, axis=[1, 2])
    elif rank_I == 3:
        inputs = tf.squeeze(inputs, axis=1)    # (None,None,K*n_filters)

    if act_f:
        inputs = act_f(inputs)

    return inputs

def highway_block(inputs, units,scope=None):
    '''
    高速网络块
    '''
    with tf.variable_scope(scope):
        H = tf.layers.dense(inputs, units, activation=tf.nn.relu,name='H')
        T = tf.layers.dense(inputs, units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0),name='T')
    return H*T+inputs*(1-T)