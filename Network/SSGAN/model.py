import tensorflow as tf
import params as par
import vlib


class Discriminator:
    def __init__(self, learning_rate, input_shape, num_classes, net='discriminator', reuse=False, X=None):
        self.learning_rate = learning_rate
        self.input_shape = [None] + input_shape
        self.num_classes = num_classes
        self.reuse = reuse
        self.net = net
        self.X = X
        self.__build_net__()
        # self.sess = session

    def __build_net__(self):
        with tf.variable_scope(self.net + '_plceholder', reuse=self.reuse):
            self.dropout = tf.placeholder(dtype=tf.float64, shape=(), name='dropout_scalar')
            if self.X is None:
                self.X = tf.placeholder(shape=self.input_shape, dtype=tf.float64, name='X')
            self.Y = tf.placeholder(dtype=tf.uint8, shape=[None, ])
            stack = None
        with tf.variable_scope(self.net + '_conv_var',reuse=tf.AUTO_REUSE):# reuse=self.reuse):
            for i in range(3):
                out = self.__conv_filtering__(i, self.X)
                if stack is None:
                    stack = out
                else:
                    stack = tf.concat([stack, out], axis=1)

        with tf.variable_scope(self.net + '_fc_layer', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('fc_w', shape=[6, self.num_classes], dtype=tf.float64)
            b = tf.get_variable('fc_b', shape=[self.num_classes, ], dtype=tf.float64)

            self.class_logits = tf.matmul(stack, W) + b
            h = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.class_logits,
                labels=tf.one_hot(self.Y, depth=self.num_classes),
            )
            self.loss = tf.reduce_mean(
                h
            )
            self.out = tf.nn.softmax(self.class_logits, name='predict')
        #self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # self.gan_logits = tf.reduce_logsumexp(self.class_logits, 1)

    def __conv_filtering__(self, cycle, tensor, net='window'):
        #print(self.reuse)
        conv = tf.layers.conv1d(
            inputs=tensor,
            filters=2,
            kernel_size=[3 + cycle],
            padding='same',
            activation=tf.nn.relu,
            name=net + '_conv_' + str(cycle),
        )

        conv = tf.nn.dropout(conv, 1 - self.dropout)
        squeeze_and_max_pool = tf.squeeze(
            tf.layers.max_pooling1d(
                conv,
                pool_size=[self.input_shape[1]],
                padding='valid',
                strides=1,
                name=net + '_maxpool1d_' + str(cycle))
        )
        return squeeze_and_max_pool

    # def train(self,input,label,dropout=0):
    #     return self.sess.run(
    #         [self.optim,self.loss],
    #         feed_dict={self.X:input,self.Y:label,self.dropout:dropout}
    #     )[1]
    #
    # def test(self,input,dropout=0):
    #     return self.sess.run(
    #         tf.argmax(self.out,axis=1),
    #         feed_dict={self.X:input,self.dropout:dropout}
    #     )
    # def test_array(self,input,dropout=0):
    #     return self.sess.run(
    #         self.out,
    #         feed_dict={self.X: input, self.dropout: dropout}
    #     )
    #
    # def test_single_text(self,input,dropout=0):
    #     result = self.sess.run(
    #         tf.argmax(self.out,axis=1),
    #         feed_dict={
    #             self.X:[input,input],
    #             self.dropout:dropout
    #         }
    #     )[0]
    #
    #     #elf.sess.close()
    #     return result


class Generator:
    def __init__(self, output_dim=par.max_length * par.embedding_dim, z=None, name=None):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            z = tf.cast(z, tf.float64)
            W1 = tf.get_variable(name="W1", shape=[z.shape[1], 1024], dtype=tf.float64)
            b1 = tf.get_variable(name="b1", shape=[1024, ], dtype=tf.float64)
            L1 = tf.nn.relu(tf.matmul(z, W1) + b1)

            W2 = tf.get_variable(name="W2", shape=[1024, 1024], dtype=tf.float64)
            b2 = tf.get_variable(name="b2", shape=[1024, ], dtype=tf.float64)
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

            W3 = tf.get_variable(name="W3", shape=[1024, output_dim], dtype=tf.float64)
            b3 = tf.get_variable(name="b3", shape=[output_dim, ], dtype=tf.float64)
            self.img = tf.tanh(
                tf.reshape(
                    tf.matmul(L2, W3) + b3,
                    shape=[z.shape[0], par.max_length, par.embedding_dim]
                )
            )
