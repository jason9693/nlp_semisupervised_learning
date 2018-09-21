import tensorflow as tf
import params as par
from Network.SSGAN import model


class GanModel:
    def __init__(self, learning_rate, input_shape, num_classes, session, ckpt_path=None):
        self.sess = session
        # if ckpt_path is not None:
        #     saver = tf.train.Saver()
        #     self.sess.run(tf.global_variables_initializer())
        #     saver.restore(self.sess, ckpt_path)
        self.supervised_flag = 0.1
        self.z_dim = par.z_dim
        self.label_holder = tf.placeholder(dtype=tf.int32,
                           shape=[par.batch_size,])
        self.label = tf.one_hot(
            self.label_holder,
            axis=1,
            depth=num_classes - 1,
            dtype=tf.float64
        )
        print(self.label.shape)
        self.G = model.Generator(
            #par.max_length,
            z = tf.random_normal(
                shape=[par.batch_size, self.z_dim]
            )
        )
        self.real_model = model.Discriminator(
            learning_rate,
            input_shape,
            num_classes,
            reuse= ckpt_path is not None
            #net='real_data',
        )

        self.fake_model = model.Discriminator(
            learning_rate,
            input_shape,
            num_classes,
            #net='fake_data',
            X=self.G.img,
            reuse=True
        )

        d_logits_pair = {
            "real": self.real_model.class_logits,
            "fake": self.fake_model.class_logits
        }
        supervised_label, unsupervised_labels = self.set_label()
        #logits_real, logits_fake = real_model.out, fake_model.out
        unsupervised_loss_pair = self.set_naive_loss(
            labels_pair= unsupervised_labels,
            logits_pair= d_logits_pair
        )
        supervised_loss_pair = self.set_naive_loss(
            labels_pair= {
                "real":supervised_label,
                "fake":unsupervised_labels["fake"]
            },
            logits_pair=d_logits_pair
        )

        feature_match = tf.reduce_mean(tf.square(self.real_model.out - self.fake_model.out))
        #regular = tf.add_n(tf.get_collection('regularizer','discriminator'),'loss')

        self.d_loss = \
            - unsupervised_loss_pair["real"] + \
            unsupervised_loss_pair["fake"] + \
            supervised_loss_pair["real"] * self.supervised_flag*10
        self.g_loss = unsupervised_loss_pair["fake"] + 0.01*feature_match

        adam_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5
        )
        rms_optim = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5
        )

        all_vars = tf.global_variables()
        self.d_optim = rms_optim.minimize(self.d_loss, var_list=[v for v in all_vars if 'discriminator' in v.name])
        self.g_optim = adam_optim.minimize(self.g_loss, var_list=[v for v in all_vars if 'generator' in v.name])
        print([v for v in all_vars if 'discriminator' in v.name])
        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess,ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def set_label(self):
        un_label_r = tf.concat([tf.ones_like(self.label), tf.zeros(shape=(par.batch_size, 1),dtype=tf.float64)], axis=1)
        un_label_f = tf.concat([tf.zeros_like(self.label), tf.ones(shape=(par.batch_size, 1),dtype=tf.float64)], axis=1)
        s_label = tf.concat([self.label, tf.zeros(shape=(par.batch_size, 1),dtype=tf.float64)], axis=1)
        return s_label, {"real":un_label_r, "fake":un_label_f}

    def set_naive_loss(self, labels_pair, logits_pair):
        real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels= labels_pair["real"] * 0.9,
                logits= logits_pair["real"]
            )
        )
        fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels= labels_pair["fake"] * 0.9,
                logits= logits_pair["fake"]
            )
        )
        return {"real": real, "fake": fake}

    def train(self,realX,realY,dropout = 0.1):
        feed_dict = {
            self.real_model.X: realX,
            self.label_holder: realY,
            self.real_model.dropout : dropout,
            self.fake_model.dropout : dropout
        }
        _,loss_d = self.sess.run([self.d_optim,self.d_loss], feed_dict = feed_dict)
        _,loss_g = self.sess.run([self.g_optim,self.g_loss], feed_dict = feed_dict)
        return loss_d,loss_g

    def test(self, input, dropout=0):
        #print(self.real_model.out.shape)
        return self.sess.run(
            tf.argmax(self.real_model.out,axis=1),
            feed_dict={self.real_model.X:input,self.real_model.dropout:dropout}
        )
