import tensorflow as tf
import numpy as np
import params as par
import tensorflow.contrib.layers as layers
import Util.tf_utils as tf_utils

class Judge:
    def __init__(self, s ,y, dropout, net="judge_net"):

        self.net = net
        self.X = s
        self.y = y
        self.dropout = dropout

        self.__build_net()


    def __build_net(self):
        with tf.device(tf_utils.gpu_mode(par.gpu)) and tf.variable_scope(self.net,reuse=tf.AUTO_REUSE):
            rpos, rneg = self.__build_r_pos_neg__(self.y)
            rs = self.__build_rs__(self.X)
            self.logits = self.L(rpos,rneg,rs)

    def L(self, rpos, rneg, rs):
        rpos = tf.expand_dims(rpos, 2)
        rneg = tf.expand_dims(rneg, 2)
        U = tf.get_variable(name='U',shape=[par.embedding_dim,par.embedding_dim], dtype=tf.float32, initializer=layers.xavier_initializer())
        rsU = tf.expand_dims(
            tf.nn.relu(tf.matmul(rs, U)),
            1
        )
        logits = tf.matmul(rsU, rpos) - tf.matmul(rsU, rneg)
        return logits


    def __build_r_pos_neg__(self, y):
        W = tf.get_variable(
            name='Wpos_neg',
            shape=[par.num_classes,par.embedding_dim],
            dtype=tf.float32
        )
        return (tf.matmul(y,W), tf.matmul(1-y,W)) #(rpos,rneg)

    def __build_rs__(self, s):
        stack = None
        with tf.variable_scope(self.net + '_conv_var',reuse=tf.AUTO_REUSE) and tf.device(tf_utils.gpu_mode(par.gpu)):
            for i in range(2):
                out = self.__conv_filtering__(i, s)
                if stack is None:
                    stack = out
                else:
                    stack = tf.concat([stack, out], axis=1)

        with tf.variable_scope(self.net + '_fc_layer', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('fc_w', shape=[800, par.embedding_dim], dtype=tf.float32, initializer=layers.xavier_initializer())
            b = tf.get_variable('fc_b', shape=[par.embedding_dim, ], dtype=tf.float32)

            return tf.matmul(stack, W) + b #activation func?

    def __conv_filtering__(self, cycle, tensor, net='window'):
        #print(self.reuse)
        conv = tf.layers.conv1d(
            inputs=tensor,
            filters=400,
            kernel_size=[3 + cycle * 2],
            padding='same',
            activation=tf.nn.relu,
            name=net + '_conv_' + str(cycle),
        )
        # conv = layers.batch_norm(conv)
        squeeze_and_max_pool = tf.squeeze(
            tf.layers.max_pooling1d(
                conv,
                pool_size=[par.max_length],
                padding='valid',
                strides=1,
                name=net + '_maxpool1d_' + str(cycle))
        )
        return squeeze_and_max_pool

    def __residual__(self, cycle, layer, tensor):
        conv = tf.layers.conv2d(
            inputs=tensor,
            filters=2 ** (3 + cycle),
            kernel_size=[3, 3],
            strides=[1, 1],
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv' + str(cycle) + '_' + str(layer),
            activation=tf.nn.relu,
            padding='same'
        )
        try:
            return conv + tensor
        except ValueError:
            tr_tens = tf.transpose(tensor, perm=[3, 0, 1, 2])
            tr_conv = tf.transpose(conv, perm=[3, 0, 1, 2])

            return tf.transpose(
                tf.concat([tr_tens, tr_tens], axis=0) + tr_conv, perm=[1, 2, 3, 0]
            )