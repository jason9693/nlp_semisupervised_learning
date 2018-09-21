import numpy as np
import tensorflow as tf
import params as par
from Network.SSVAE.vae import VariationalAutoencoder
from Network.baseNet import textCNN as net
import Util.dgm_utils as util
import Util.textcnn_util as text_util
import Util.tf_utils as tf_utils
import pandas as pd
from gensim.models.word2vec import Word2Vec as w2v
import pprint

class M2:
    def __init__(self,
                 dim_x, dim_y, dim_z, dim_h=500,
                 l2_loss=0.1,
                 learning_rate=1e-3,
                 sess=None,
                 ckpt_path = None
                 ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_h = dim_h

        self.l2 = l2_loss
        self.learning_rate = learning_rate

        self.__build_net__()

        self.xy_recon_x = self.network(self.flat_x, self.y, self.rand_labeled) # input : (x,y) out: reconstruct x
       # self.x_recon_x = self.network(self.flat_unlabeled_x,y=None, z=self.rand_unlabeled) # input : (x) out: reconstruct x

        self.sess = sess

        self.classify = self.classifier(x=self.x,y=self.naive_y)
        self.alpha = tf.placeholder(dtype=tf.float32, shape=(), name='alpha')
        self.unlabeled_loss = tf.reduce_sum(self.U(self.flat_unlabeled_x))
        self.labeled_loss = tf.reduce_sum(self.L(self.flat_x,self.y, self.rand_labeled))
        self.loss =  self.unlabeled_loss + self.alpha * tf.reduce_mean(self.classify.h) + self.labeled_loss


        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess,ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        print("Variables list is follow:")
        pprint.pprint(tf.global_variables())

    def encoder(self, x, name='encoder'):
        with tf.variable_scope(name, reuse= tf.AUTO_REUSE):
            L1 = self.__dense__(x, x.shape[1], self.dim_h, name=name + '_L1')
            L1 = tf_utils.leaky_relu(L1, 0.01)
            L2 = self.__dense__(L1, self.dim_h, self.dim_z * 2, name=name + '_L2')

            mu = L2[:, :self.dim_z]
            var = L2[:, self.dim_z:]
            var = tf.nn.softplus(var)
        return mu, var

    def decoder(self, z, name='decoder'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            L1 = self.__dense__(z, z.shape[1], self.dim_h, name=name + '_L1')
            L1 = tf_utils.leaky_relu(L1, 0.01)
            L2 = self.__dense__(L1, self.dim_h, self.dim_x, name=name + '_L2')
        return tf.tanh(L2) #w2v의 out은 -1~1 사이


    def __dense__(self, x, input_dim, output_dim, name='net'):
        with tf.name_scope(name):
            W = tf.get_variable(shape=[input_dim, output_dim], dtype=tf.float32, name=name+'_W')
            b = tf.get_variable(shape=[output_dim], dtype=tf.float32, name=name+'_b')
            return tf.matmul(x, W) + b

    def __build_net__(self):
        self.unlabeled_x = tf.placeholder(shape=[None, par.max_length, par.embedding_dim],dtype=tf.float32)
        self.x = tf.placeholder(shape=(None, par.max_length, par.embedding_dim),dtype=tf.float32)

        self.flat_x = tf.layers.flatten(self.x)
        self.flat_unlabeled_x = tf.layers.flatten(self.unlabeled_x) # 인코더에는 flatten한 input이 요구됨.

        self.naive_y = tf.placeholder(shape=(None,),dtype=tf.uint8)
        self.y = tf.one_hot(self.naive_y,par.num_classes,dtype=tf.float32)

        self.rand_labeled = tf.placeholder(shape=(None, self.dim_z),dtype=tf.float32)
        self.rand_unlabeled = tf.placeholder(shape=(None, self.dim_z), dtype=tf.float32)



    def network(self, x, y, z):
        if y is not None:
            x = tf.concat([x, y], axis=1)

        z_mu, z_var = self.encoder(x)

        z = z_var * z + z_mu  # TODO: shape 문제
        kld = util.kld(z, (z_mu, z_var))

        if y is not None:
            z = tf.concat([z, y], axis=1)
        x_recon = self.decoder(z)
        return (x_recon ,kld)

    def classifier(self, x, y, name = 'classifier'):
        cnn = net.TextCNN(
            input_shape=[par.max_length, par.embedding_dim],
            learning_rate=self.learning_rate,
            num_classes=par.num_classes,
            net=name,
            X=x,
            Y=y
        )
        cnn.dropout = 0
        return cnn


    def L(self,x, y, z):
        recon_x, kld = self.network(x,y,z)
        likelihood = util.binary_cross_entropy(tf.sigmoid(recon_x),tf.sigmoid(x))

        y_prior = (1. / self.dim_y) * tf.ones_like( y )
        log_prior_y = tf.nn.softmax_cross_entropy_with_logits( logits= y_prior, labels= y ) #TODO: logits,labels seq

        elbo = tf.reduce_mean(likelihood ,axis=1) + log_prior_y - kld
        #loss = tf.reduce_logsumexp(elbo + 1e-8 )
        return - elbo

    def U(self, x):
        left_temp = []
        logits = self.classifier(self.unlabeled_x,None).class_logits
        for label in range(self.dim_y):
            y = tf.zeros_like(self.flat_unlabeled_x) #TODO: fake y implement
            temp_y = y[:,label]
            temp_y = tf.zeros_like(temp_y, dtype=tf.uint8) + label
            y = tf.one_hot(temp_y,depth=self.dim_y)

            left_temp.append( -(self.L(x,y,self.rand_unlabeled) * tf.sigmoid(logits[:,label]) ))

        return -(tf.reduce_sum(left_temp,axis=0) - tf.reduce_sum(tf.sigmoid(logits) * tf.log( tf.sigmoid(logits) + 1e-8 ),axis=1))


    def train(self, labeled, unlabeled, alpha):
        labeled_x, labeled_y = labeled
        unlabeled_x = unlabeled
        np.random.seed(777)
        return self.sess.run([self.optim, self.loss, self.classify.loss, self.labeled_loss, self.unlabeled_loss],
                             feed_dict={
                                 self.x: labeled_x,
                                 self.naive_y: labeled_y,
                                 self.rand_labeled: np.random.normal(0,1,(labeled_x.shape[0],self.dim_z)),
                                 self.rand_unlabeled: np.random.normal(0,1,(unlabeled_x.shape[0],self.dim_z)),
                                 self.unlabeled_x: unlabeled_x,
                                 self.alpha: alpha
                             })

    def infer(self, x):
        return self.sess.run(
            tf.argmax(self.classifier(x=self.unlabeled_x,y=None).class_logits,axis=1),
            feed_dict={self.unlabeled_x: x}
        )

    def eval(self, x, y):

        pred_y =  self.sess.run(
            tf.argmax(self.classifier(x=self.x,y=None).class_logits,axis=1), feed_dict = {self.x: x})

        correct = 0
        for i in range(y.shape[0]):
            if y[i]==pred_y[i]:
                correct += 1

        return correct / y.shape[0]


# class StackedDGM(M2):
#     def __init__(self,
#                  dim_x, dim_y, dim_z, dim_h = 500,
#                  l2_loss = 0.1,
#                  learning_rate=1e-3,
#                  vae_path = '../checkpoints/model_VAE_0.0003-33_1535471737.939126.cpkt'
#                  ):
#         super(StackedDGM, self).__init__(dim_x,dim_y,dim_z,dim_h)
#         self.dim_x = dim_x
#         self.dim_y = dim_y
#         self.dim_z = dim_z
#         self.dim_h = dim_h
#
#         self.l2 = l2_loss
#         self.learning_rate = learning_rate
#         self.vae_path = vae_path
#         self.vae = VariationalAutoencoder( dim_x = dim_x, dim_z = dim_z )
#
#         x = tf.placeholder(dtype=tf.float32, shape = [None, dim_x])
#         mu,sig = self.vae.encode(x)
#         x_recon = self.__build__net__(mu, sig)
#     pass
#
#
#     def __build__net__(self, mu, sigma, name='decoder'):
#         with tf.name_scope(name):
#             z = tf.random_normal(mean=0, stddev=1, shape=[None, self.dim_z])
#             z = sigma * z + mu
#         L1 = self.__dense__(z, z.shape, self.dim_h, name=name + '_L1')
#         L2 = self.__dense__(L1, self.dim_h, self.dim_x, name=name + '_L2')
#         return super(StackedDGM, self).encoder(L2,name='M2_encoder')
#
#     def classify(self, x, name = 'classifier'):
#         with self.vae.session:
#             self.vae.saver.restore(self.vae.session, self.vae_path)
#             x, _ = self.vae.encode(x)
#
#         cnn = net.TextCNN(
#             input_shape=self.dim_z,
#             learning_rate=self.learning_rate,
#             num_classes=par.num_classes,
#             net=name
#         )
#         return cnn.class_logits


if __name__ == '__main__':
    tf.set_random_seed(777)
    dgm = M2(
        dim_x= par.max_length * par.embedding_dim,
        dim_y= par.num_classes,
        dim_z=50,
        # ckpt_path='test_dgm.ckpt',
        sess=tf.Session()
    )

    dataframe = pd.read_csv('../../csv/tagged_csv/everytime_tagged.csv')
    model = '../../w2v_model/everytime_campusPick_kowiki.model'


    train_y, train_x, test_y, test_x, _ = \
        text_util.div_dataset(dataframe,w2v_model=model, embedding_dim=par.embedding_dim, train_size=400)

    print(test_y.shape)

    dgm.train(labeled=(test_x, test_y), unlabeled=train_x, alpha=1)

    dgm.infer(x=test_x)

