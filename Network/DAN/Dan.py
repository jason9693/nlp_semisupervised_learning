import tensorflow as tf
import params as par
from Network.baseNet import textCNN as net
from Network.DAN.JudgeNet import Judge
from pprint import pprint
import tensorflow.contrib.layers as layers

class DAN:
    def __init__(self, learning_rate, input_shape, num_classes, sess, ckpt_path=None, mode='w2v'):
        self.sess = sess
        self.learning_rate = learning_rate
        self.input_shape= input_shape

        if mode is 'w2v':
            self.labeled_x = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape, name='sentence')
            self.unlabeled_x = tf.placeholder(dtype=tf.float32, shape=[None]+input_shape, name='sentence')
            self.unlabeled_s = self.unlabeled_x
            self.labeled_s = self.labeled_x
        else:
            self.labeled_x = tf.placeholder(dtype=tf.int32, shape=[None, input_shape[0]], name='sentence')
            self.unlabeled_x = tf.placeholder(dtype=tf.int32, shape=[None, input_shape[0]], name='sentence')
            self.labeled_s = self.__embedding_s__(
                self.labeled_x
            )
            self.unlabeled_s = self.__embedding_s__(
                self.unlabeled_x
            )
        self.label_holder = tf.placeholder(dtype=tf.int32,
                           shape=[None,])
        self.dropout = tf.placeholder(shape=(), dtype=tf.float32)

        self.label = tf.one_hot(
            self.label_holder,
            axis=1,
            depth=num_classes,
            dtype=tf.float32
        )

        self.P = net.TextCNN(
            learning_rate,
            input_shape,
            num_classes,
            net='Predictor',
            X=self.unlabeled_s
        )
        self.P.dropout = self.dropout
        self.__build_net__()


        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess,ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())


    def __embedding_s__(self, x):
        char_len = 251
        with tf.variable_scope('embedding_s', reuse=tf.AUTO_REUSE):
            try:
                char_embedding = tf.get_variable('char_embedding', [char_len, self.input_shape[1]], initializer=layers.xavier_initializer())
            except:
                return
        return tf.nn.embedding_lookup(char_embedding, x)

    def __build_net__(self):
        self.P_j = self.JudgeNet(x=self.unlabeled_s)
        self.J = self.JudgeNet(x=self.labeled_s, y=self.label)

        real_label = tf.ones_like(self.P_j.logits)
        fake_label = tf.zeros_like(self.P_j.logits)

        D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.P_j.logits, labels=fake_label)
        D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.J.logits, labels=real_label)

        self.D_loss = tf.reduce_mean(D_loss_fake + D_loss_real)

        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.P_j.logits, labels=real_label)
        self.G_loss = tf.reduce_mean(self.G_loss)

        variables = tf.global_variables()

        P_varlist = [v for v in variables if 'Predictor' in v.name] #+ [v for v in variables if 'embedding_s' in v.name]
        J_varlist = [v for v in variables if 'Judge' in v.name]# + [v for v in variables if 'embedding_s' in v.name]
        self.P_optim = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        ).minimize(self.G_loss, var_list=P_varlist)

        self.J_optim = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate * 2
        ).minimize(self.D_loss, var_list=J_varlist)

        print('P_var: ')
        pprint(P_varlist)
        print('-----------')
        print('J var: ')
        pprint(J_varlist)

    def JudgeNet(self,x,y= None):
        if y is None:
            # y = tf.argmax(self.P.class_logits,1)
            # y = tf.one_hot(y, axis=1, dtype= tf.float32, depth=par.num_classes)
            #y = self.P.class_logits
            y  = self.P.out

        J = Judge(
            s=x,
            y=y,
            dropout=self.dropout,
            net='Judge'
        )

        return J

    def train(self, unlabeled, labeled=None, dropout=0.1):
        if labeled is None:
            return self.sess.run([self.P_optim, self.G_loss], feed_dict = {self.unlabeled_x : unlabeled, self.dropout: dropout})
        else:
            s, y = labeled
            return self.sess.run(
                [self.J_optim, self.D_loss],
                feed_dict = {self.labeled_x : s, self.unlabeled_x: unlabeled, self.label_holder: y, self.dropout: dropout}
            )

    def eval(self, labeled):
        s, y = labeled
        predicts,logits = \
            self.sess.run( [tf.argmax(self.P.out, axis= 1),self.P.class_logits] , feed_dict = {self.unlabeled_x: s, self.dropout: 0.0} )
        #print(logits)
        accuracy = 0
        for i in range(len(predicts)):
            if predicts[i] == y[i]:

                accuracy += 1

        return accuracy / len(predicts)

    def infer(self,x):
        return self.sess.run(tf.argmax(self.P.out,axis=1),feed_dict={self.unlabeled_x:x, self.dropout: 0.0})
#
# class Predictor:
#     def __init__(self, input_shape, num_classes, net='Predictor', reuse=False, X=None, Y = None):
#         self.input_shape = [None] + input_shape
#         self.num_classes = num_classes
#         self.reuse = reuse
#         self.net = net
#         self.X = X
#         self.Y = Y
#         self.__build_net__()
#
#     def __build_net__(self):
#
#         pass
#
#     def __conv_filtering__(self, cycle, tensor, net='window'):
#         #print(self.reuse)
#         conv = tf.layers.conv1d(
#             inputs=tensor,
#             filters=2,
#             kernel_size=[3 + cycle],
#             padding='same',
#             activation=tf.nn.relu,
#             name=net + '_conv_' + str(cycle),
#         )
#
#         conv = layers.batch_norm(conv)
#         squeeze_and_max_pool = tf.squeeze(
#             tf.layers.max_pooling1d(
#                 conv,
#                 pool_size=[self.input_shape[1]],
#                 padding='valid',
#                 strides=1,
#                 name=net + '_maxpool1d_' + str(cycle))
#         )
#         return squeeze_and_max_pool
