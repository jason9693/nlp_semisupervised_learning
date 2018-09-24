import tensorflow as tf
import tensorflow.contrib.layers as layers
import Util.tf_utils as tf_utils

class TextCNN:
    def __init__(self, learning_rate, input_shape, num_classes, net='discriminator', reuse=False, X=None, Y = None):
        self.learning_rate = learning_rate
        self.input_shape = [None] + input_shape
        self.num_classes = num_classes
        self.reuse = reuse
        self.net = net
        self.X = X
        self.Y = Y
        self.__build_net__()
        # self.sess = session

    def __build_net__(self):
        with tf.variable_scope(self.net + '_plceholder', reuse=self.reuse):
            self.dropout = 0.0
            #tf.placeholder(dtype=tf.float32, shape=(), name='dropout_scalar')
            if self.X is None:
                self.X = tf.placeholder(shape=self.input_shape, dtype=tf.float64, name='X')
            if self.Y is None:
                self.Y = tf.placeholder(dtype=tf.uint8, shape=[None, ])
            stack = None
        with tf.variable_scope(self.net + '_conv_var',reuse=tf.AUTO_REUSE):# reuse=self.reuse):
            for i in range(2):
                out = self.__conv_filtering__(i, self.X)
                if stack is None:
                    stack = out
                else:
                    stack = tf.concat([stack, out], axis=1)

        with tf.variable_scope(self.net + '_fc_layer', reuse=tf.AUTO_REUSE):
            W1 = tf.get_variable('fc_w1',shape=[200,50], dtype=tf.float32)
            b1 = tf.get_variable('fc_b1',shape=[50],dtype=tf.float32)
            stack = tf.matmul(stack,W1)+ b1
            stack = tf_utils.leaky_relu(stack,0.01)

            W = tf.get_variable('fc_w', shape=[50, self.num_classes], dtype=tf.float32)
            b = tf.get_variable('fc_b', shape=[self.num_classes, ], dtype=tf.float32)

            self.class_logits = tf.matmul(stack, W) + b
            self.h = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.class_logits,
                labels=tf.one_hot(self.Y, depth=self.num_classes),
            )
            self.loss = tf.reduce_mean(
                self.h
            )
            self.out = tf.nn.softmax(self.class_logits, name='predict')
            #self.out = tf.nn.tanh(self.class_logits) * 0.5 + 0.5
        #self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # self.gan_logits = tf.reduce_logsumexp(self.class_logits, 1)

    def __conv_filtering__(self, cycle, tensor, net='window'):
        #print(self.reuse)
        conv = tf.layers.conv1d(
            inputs=tensor,
            filters=100,
            kernel_size=[3 + cycle],
            padding='same',
            activation=tf_utils.leaky_relu,
            name=net + '_conv_' + str(cycle),
        )
        #conv = tf.nn.dropout(conv, 1 - self.dropout)
        # conv = layers.batch_norm(conv)
        squeeze_and_max_pool = tf.squeeze(
            tf.layers.max_pooling1d(
                conv,
                pool_size=[self.input_shape[1]],
                padding='valid',
                strides=1,
                name=net + '_maxpool1d_' + str(cycle))
        )
        return squeeze_and_max_pool