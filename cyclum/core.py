"""
This module implements the scae core.
Current version is a fast implementation
"""

#sphinx-build -b html rst html
#sphinx-apidoc -o . .

import numpy as np
import tensorflow as tf
import time
from sklearn.decomposition import PCA


class _BaseCyclum:
    """
    The base class for all the realizations.
    All this class knows is math.
    """
    __slots__ = ["Y_value", "N", "P", "Q", "Y", "X", "Y2", "isbuilt"]

    def __init__(self, Y, Q, ispreload=True):
        """
        :type Y: numpy matrix
        :param Y:
        :type q: int
        :param q: dimension of embedding (pseudotime)
        """
        self.Y_value = Y
        self.N, self.P = Y.shape
        self.Q = Q
        if ispreload:
            self.Y = tf.constant(Y, dtype=tf.float32)
        else:
            self.Y = tf.placeholder([None, self.P])
        self.isbuilt = False

    def linear_encoder(qs):
        """
        declare a linear encoder

        :param qs: the index of rows in embedding to generate
        :return: an dict in a list; the sum of these list is the required encoder configuration for build()
        """
        return [dict(type='linear', qs=qs)]

    def nonlinear_encoder(qs, hidden_qs):
        """
        declare a nonlinear encoder

        :param qs: the index of rows in embedding to generate
        :return: an dict in a list; the sum of these list is the required decoder configuration for build()
        """
        return [dict(type='nonlinear', qs=qs, hidden_qs=hidden_qs)]

    def linear_decoder(qs):
        """
        declare a linear decoder

        :param qs: the index of rows in embedding to be used
        :return: an dict in a list; the sum of these list is the required input of build()
        """
        return [dict(type='linear', qs=qs)]

    def circular_decoder(qs):
        """
        declare a circular decoder

        :param qs: the index of rows in embedding to be used
        :return: an dict in a list; the sum of these list is the required input of build()
        """
        return [dict(type='circular', qs=qs)]

    def _make_linear_layer(Z, Q):
        """
        make a linear layer

        :param Z: the input tensor
        :param Q: dimension of the output
        :return: the output tensor of this layer
        """
        W = tf.Variable(tf.random_normal([int(Z.shape[-1]), Q]) / 4, name='W')
        b = tf.Variable(tf.zeros([1, Q]), name='b')
        print(W)
        return tf.add(Z @ W, b, name='Z')

    def _make_nonlinear_layer(Z, Q):
        """
        make a nonlinear layer

        :param Z: the input tensor
        :param Q: dimension of the output
        :return: the output tensor of this layer
        """
        W = tf.Variable(tf.random_normal([int(Z.shape[-1]), Q]), name='W')
        b = tf.Variable(tf.zeros([1, Q]), name='b')
        print(W)
        return tf.tanh(Z @ W + b, name='Z')

    def _make_circular_layer(Z, Q):
        """
        make a circular layer

        :param Z: the input tensor
        :param Q: dimension of the output
        :return: the output tensor of this layer
        """
        assert Z.shape[1] == 1
        W = tf.Variable(tf.random_normal([3, Q]), name='W')
        temp = tf.concat([tf.cos(Z + i * 2 * np.pi / 3) for i in range(3)], 1)
        return tf.matmul(temp, W, name='Z')

    def _make_nonlinear_encoder(Z, Q, hidden_qs):
        """
        sum up the nonlinear layers to make a nonlinear encoder

        :param Z: the input tensor
        :param Q: the final output dimension
        :param hidden_qs: the hidden layer dimensions
        :return: the output tensor of this encoder
        """
        temp = Z
        for i, q in enumerate(hidden_qs):
            with tf.name_scope('layer' + str(i)):
                temp = _BaseCyclum._make_nonlinear_layer(temp, q)
        with tf.name_scope('output'):
            return _BaseCyclum._make_linear_layer(temp, Q)

    def _make_linear_encoder(Z, Q):
        """
        Use one linear layer as a linear encoder.
        No need to have hidden layers due to the property of linear transformation.

        :param Z: the input tensor
        :param Q: the output dimension
        :return: the output tensor of this encoder
        """
        with tf.name_scope('output'):
            return _BaseCyclum._make_linear_layer(Z, Q)

    def _make_linear_decoder(Z, P):
        """
        Use one linear layer as a linear decoder.
        No need to have hidden layers due to the property of linear transformation.

        :param Z: the input tensor
        :param P: the output dimension
        :return: the output tensor of this encoder
        """
        with tf.name_scope('output'):
            return _BaseCyclum._make_linear_layer(Z, P)

    def _make_circular_decoder(Z, P):
        """
        Use one circular layer as a linear decoder.

        :param Z: the input tensor
        :param P: the output dimension
        :return: the output tensor of this encoder
        """
        with tf.name_scope('output'):
            return _BaseCyclum._make_circular_layer(Z, P)

    def _make_encoder(Y, Q, encoder):
        """
        make the full encoder

        :param Y: the input tensor
        :param Q: the output dimension
        :param encoder: the encoder configuration
        :return: the output tensor of the full encoder
        """
        temp = [tf.zeros([tf.shape(Y)[0]]) for i in range(Q)]
        for i, component in enumerate(encoder):
            with tf.name_scope('encoder' + str(i)):
                if component['type'] == 'linear':
                    res = _BaseCyclum._make_linear_encoder(Y, len(component['qs']))
                    for j, q in enumerate(component['qs']):
                        temp[q] += res[:, j]
                elif component['type'] == 'nonlinear':
                    res = _BaseCyclum._make_nonlinear_encoder(Y, len(component['qs']), component['hidden_qs'])
                    for j, q in enumerate(component['qs']):
                        temp[q] += res[:, j]
                else:
                    assert False  # You should never get here
        return tf.stack(temp, axis=1, name='neck')

    def _make_decoder(X, P, decoder):
        """
        make the full decoder

        :param X: the input tensor
        :param P: the output dimension
        :param decoder: the decoder configuration
        :return: the output tensor of the full decoder
        """
        temp = []
        for i, component in enumerate(decoder):
            with tf.name_scope('decoder' + str(i)):
                if component['type'] == 'linear':
                    temp.append(_BaseCyclum._make_linear_decoder(tf.gather(X, component['qs'], axis=1), P))
                elif component['type'] == 'circular':
                    temp.append(_BaseCyclum._make_circular_decoder(tf.gather(X, component['qs'], axis=1), P))
                else:
                    assert False
        return tf.add_n(temp)

    def build(self, encoder, decoder):
        """
        build the model

        :param encoder: encoder configuration
        :param decoder: decoder configuration
        :return: None
        """
        self.X = _BaseCyclum._make_encoder(self.Y, self.Q, encoder)
        self.Y2 = _BaseCyclum._make_decoder(self.X, self.P, decoder)

    def train(self):
        """
        Train the model. To be implemented in derived classes.
        """
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

class PreloadCyclum2(_BaseCyclum):
    def __init__(self, Y):
        super().__init__(Y, 2)
        encoder = _BaseCyclum.nonlinear_encoder([0], [30, 20]) + _BaseCyclum.linear_encoder([1])
        decoder = _BaseCyclum.circular_decoder([0]) + _BaseCyclum.linear_decoder([1])
        self.build(encoder, decoder)

    def _get_initial_value(self, n_candidate=5):
        """
        Get initial value by running on the first few PCs.

        :param n_candidate: number of PCs.
        :return: proposed initial value.
        """
        pc = PCA(n_components=10, copy=True, whiten=False, svd_solver="auto",
                 tol=0.0, iterated_power="auto", random_state=None).fit_transform(self.Y_value)
        spc = pc / np.std(pc, axis=0)

        unit_score_1d = []
        uniform_score_1d = []
        ind_1d = [(i, j) for i in range(n_candidate) for j in range(i)]
        for i, j in ind_1d:
                temp = np.sqrt(spc[:, i] ** 2 + spc[:, j] ** 2)
                temp = temp / np.mean(temp)
                unit_score_1d.append(np.mean(np.abs(temp - 1)))

                temp = np.angle(spc[:, i] + spc[:, j] * 1j)
                temp.sort()
                diff = np.append(np.diff(temp), temp[0] + 2 * np.pi - temp[-1])
                uniform_score_1d.append(np.std(diff))

        min_max_normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        unit_score_1d = min_max_normalize(np.array(unit_score_1d))
        uniform_score_1d = min_max_normalize(np.array(uniform_score_1d))

        final_score_1d = unit_score_1d + uniform_score_1d
        ind = ind_1d[np.argmin(final_score_1d).item()]

        temp = np.sqrt(spc[:, ind[0]] ** 2 + spc[:, ind[1]] ** 2)
        return spc[:, ind] / np.mean(temp)

    def train(self):
        """
        train the model

        :return:
        """

        graph = tf.get_default_graph()

        paragon = self._get_initial_value()
        cossin = lambda x: tf.concat([tf.cos(x), tf.sin(x)], 1)
        Z_circular_pre_train = cossin(tf.gather(self.X, [0], axis=1))

        pretrain_loss = tf.nn.l2_loss(Z_circular_pre_train - paragon) / self.N
        pretrain_var_names = ['encoder0/layer0/W', 'encoder0/layer0/b',
                              'encoder0/layer1/W', 'encoder0/layer1/b',
                              'encoder0/output/W', 'encoder0/output/b']
        pretrain_var_list = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)[0] for name in pretrain_var_names]
        #self.pretrain_var_list = [self.tf_W_circular1, self.tf_b_circular1, self.tf_W_circular2, self.tf_b_circular2,
                                  #self.tf_W_circular3, self.tf_b_circular3]

        midtrain_var_names = ['decoder0/output/W', 'decoder1/output/W', 'decoder1/output/b']
        midtrain_var_list = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)[0] for name in midtrain_var_names]
        #self.midtrain_var_list = [self.tf_V0, self.tf_V1, self.tf_c]

        R = ((tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'encoder0/layer0/W')[0]) +
              tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'encoder0/layer1/W')[0]) +
              tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'encoder0/output/W')[0])
              ) / 10 +
             (tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'encoder1/output/W')[0]) +
              tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'decoder0/output/W')[0]) +
              tf.nn.l2_loss(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'decoder1/output/W')[0])
              )
             )

        L = tf.reduce_sum((self.Y - self.Y2) ** 2) / (2 * self.N)

        pretrain_burnin = tf.train.AdamOptimizer(50e-3).minimize(pretrain_loss, var_list=pretrain_var_list)
        pretrain_train = tf.train.AdamOptimizer(1e-3).minimize(pretrain_loss, var_list=pretrain_var_list)

        midtrain_burnin = tf.train.AdamOptimizer(50e-3).minimize(L + R * 1.0, var_list=midtrain_var_list)
        midtrain_train = tf.train.AdamOptimizer(1e-3).minimize(L + R * 1.0, var_list=midtrain_var_list)

        #tf_burnin = tf.train.AdamOptimizer(50e-3).minimize(self.tf_L + self.tf_R * 0.2)
        final_train = tf.train.AdamOptimizer(2e-4).minimize(L + R * 1.0)
        final_refine = tf.train.AdamOptimizer(5e-5).minimize(L + R * 1.0)

        t0 = time.time()
        #saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        def unit_train(opt, n_step, losses, sess):
            t1 = time.time()
            for i in range(n_step):
                sess.run(opt)
                if (i + 1) % 2000 == 0:
                    print('step %5d: loss ' % (i + 1), end="")
                    print(self.sess.run(losses), end="")
                    print(' time %.2f' % (time.time() - t1))
                    t1 = time.time()

        print("pretrain burnin")
        unit_train(pretrain_burnin, 2000, [pretrain_loss, L, R], self.sess)

        print("pretrain train")
        unit_train(pretrain_train, 4000, [pretrain_loss, L, R], self.sess)
        
        print("midtrain burnin")
        unit_train(midtrain_burnin, 2000, [pretrain_loss, L, R], self.sess)

        print("midtrain train")
        unit_train(midtrain_train, 4000, [pretrain_loss, L, R], self.sess)

        print("finaltrain train")
        unit_train(final_train, 6000, [pretrain_loss, L, R], self.sess)

        print("finaltrain refine")
        unit_train(final_refine, 10000, [pretrain_loss, L, R], self.sess)

        self.pseudotime = self.sess.run(self.X)
        self.rotation = self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'decoder0/output/W')[0])

        print('Full time %.2f' % (time.time() - t0))

        return self.pseudotime, self.rotation


class PreloadCyclum:
    """
    The core of Cyclum.
    The data is preloaded into graphic memory to run it as fast as possible.
    Before fit()

    :type Y: numpy matrix
    :param Y: each row is a cell and each column is a gene.
    :param q_circular: number of
    :param q_linear:
    :param seed: random seed
    :type verbose: bool
    :param verbose: if True, show the training progress
    """
    def __init__(self, Y, q_circular=3, q_linear=0, seed=0, verbose=True):
        """
        Initialize the model.

        """
        self.verbose = verbose

        tf.set_random_seed(seed)
        self.q_circular = q_circular
        self.q_linear = q_linear
        self.q = q_linear + q_circular

        n, d = Y.shape

        self.Y = Y

        self.tf_X = tf.constant(Y, dtype=tf.float32)

        # Linear Part
        self.tf_W = tf.Variable(tf.random_normal([d, self.q_linear]) / 4)
        self.tf_b = tf.Variable(tf.zeros([1, self.q_linear]))

        self.tf_E = (self.tf_X @ self.tf_W + self.tf_b)

        # Circular part
        self.tf_W_circular1 = tf.Variable(tf.random_normal([d, 30]))
        self.tf_b_circular1 = tf.Variable(tf.zeros([1, 30]))

        self.tf_E_circular1 = tf.tanh(self.tf_X @ self.tf_W_circular1 + self.tf_b_circular1)

        self.tf_W_circular2 = tf.Variable(tf.random_normal([30, 20]))
        self.tf_b_circular2 = tf.Variable(tf.zeros([1, 20]))

        self.tf_E_circular2 = tf.tanh(self.tf_E_circular1 @ self.tf_W_circular2 + self.tf_b_circular2)

        self.tf_W_circular3 = tf.Variable(tf.random_normal([20, 1]))
        self.tf_b_circular3 = tf.Variable(tf.zeros([1, 1]))

        self.tf_E_circular = self.tf_E_circular2 @ self.tf_W_circular3 + self.tf_b_circular3

        # slice
        self.tf_E_linear = self.tf_E

        tf_phase = lambda x, num: tf.concat([tf.cos(x + i * 2 * np.pi / num) for i in range(num)], 1)
        tf_cossin = lambda x: tf.concat([tf.cos(x), tf.sin(x)], 1)

        if q_circular == 2:
            self.tf_Z_circular = tf_cossin(self.tf_E_circular)
        else:
            self.tf_Z_circular = tf_phase(self.tf_E_circular, 3)

        self.tf_Z_linear = self.tf_E_linear

        # decode
        self.tf_D_circular = self.tf_Z_circular
        self.tf_D_linear = self.tf_Z_linear

        self.tf_V0 = tf.Variable(tf.random_normal([self.q_circular, d]))
        self.tf_V1 = tf.Variable(tf.random_normal([self.q_linear, d]))

        self.tf_c = tf.Variable(tf.zeros([1, d]))

        self.tf_X_r_circular = self.tf_D_circular @ self.tf_V0
        self.tf_X_r = self.tf_X_r_circular + self.tf_D_linear @ self.tf_V1 + self.tf_c

        self.tf_R = (tf.nn.l2_loss(self.tf_W_circular1) + tf.nn.l2_loss(self.tf_W_circular2) + tf.nn.l2_loss(self.tf_W_circular3)) / 10 + tf.nn.l2_loss(self.tf_W) + tf.nn.l2_loss(self.tf_V1) + tf.nn.l2_loss(self.tf_V0)

        self.tf_L = tf.reduce_sum((self.tf_X - self.tf_X_r) ** 2) / (2 * n)

        # pretrain part
        paragon = self._get_initial_value()
        self.tf_Z_circular_pre_train = tf_cossin(self.tf_E_circular)

        self.tf_pretrain_loss = tf.nn.l2_loss(self.tf_Z_circular_pre_train - paragon) / n
        self.pretrain_var_list = [self.tf_W_circular1, self.tf_b_circular1, self.tf_W_circular2, self.tf_b_circular2,
                                  self.tf_W_circular3, self.tf_b_circular3]

        self.midtrain_var_list = [self.tf_V0, self.tf_V1, self.tf_c]

    def _get_initial_value(self, n_candidate=5):
        """
        Get initial value by running on the first few PCs.

        :param n_candidate: number of PCs.
        :return: proposed initial value.
        """
        pc = PCA(n_components=10, copy=True, whiten=False, svd_solver="auto",
                 tol=0.0, iterated_power="auto", random_state=None).fit_transform(self.Y)
        spc = pc / np.std(pc, axis=0)

        unit_score_1d = []
        uniform_score_1d = []
        ind_1d = [(i, j) for i in range(n_candidate) for j in range(i)]
        for i, j in ind_1d:
                temp = np.sqrt(spc[:, i] ** 2 + spc[:, j] ** 2)
                temp = temp / np.mean(temp)
                unit_score_1d.append(np.mean(np.abs(temp - 1)))

                temp = np.angle(spc[:, i] + spc[:, j] * 1j)
                temp.sort()
                diff = np.append(np.diff(temp), temp[0] + 2 * np.pi - temp[-1])
                uniform_score_1d.append(np.std(diff))

        min_max_normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        unit_score_1d = min_max_normalize(np.array(unit_score_1d))
        uniform_score_1d = min_max_normalize(np.array(uniform_score_1d))

        final_score_1d = unit_score_1d + uniform_score_1d
        ind = ind_1d[np.argmin(final_score_1d).item()]

        #if self.verbose:
        #    print(ind)
        #    print(ind_1d)
        #    print(unit_score_1d)
        #    print(uniform_score_1d)

        temp = np.sqrt(spc[:, ind[0]] ** 2 + spc[:, ind[1]] ** 2)
        return spc[:, ind] / np.mean(temp)

    def fit(self):
        """
        Fits the model and give the inferred pseudotime, and also its relationship to the gene expression.
        These outputs are essential for downstream analysis.

        :return:
            pseudotime: the pseudo-time for each cell, in the same order as the input, in a [0, 2\pi] scale.
            rotation: the rotation matrix
        """
        tf_pretrain_burnin = tf.train.AdamOptimizer(50e-3).minimize(self.tf_pretrain_loss, var_list=self.pretrain_var_list)
        tf_pretrain_train = tf.train.AdamOptimizer(1e-3).minimize(self.tf_pretrain_loss, var_list=self.pretrain_var_list)

        tf_midtrain_burnin = tf.train.AdamOptimizer(50e-3).minimize(self.tf_L + self.tf_R * 1.0,
                                                                    var_list=self.midtrain_var_list)
        tf_midtrain_train = tf.train.AdamOptimizer(1e-3).minimize(self.tf_L + self.tf_R * 1.0,
                                                                    var_list=self.midtrain_var_list)

        #tf_burnin = tf.train.AdamOptimizer(50e-3).minimize(self.tf_L + self.tf_R * 0.2)
        tf_train = tf.train.AdamOptimizer(2e-4).minimize(self.tf_L + self.tf_R * 1.0)
        tf_refine = tf.train.AdamOptimizer(5e-5).minimize(self.tf_L + self.tf_R * 1.0)

        t1 = t0 = time.time()
        #saver = tf.train.Saver()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print("pretrain burnin")
        for i in range(2000):
            self.sess.run(tf_pretrain_burnin)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        print("pretrain train")
        for i in range(4000):
            self.sess.run(tf_pretrain_train)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        print("midtrain burnin")
        for i in range(2000):
            self.sess.run(tf_midtrain_burnin)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        print("midtrain train")
        for i in range(4000):
            self.sess.run(tf_midtrain_train)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        print("finaltrain train")
        for i in range(6000):
            self.sess.run(tf_train)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        print("finaltrain refine")
        for i in range(10000):
            self.sess.run(tf_refine)
            if (i + 1) % 2000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(self.sess.run([self.tf_pretrain_loss, self.tf_L, self.tf_R]), end="")
                print(' time %.2f' % (time.time() - t1))
                t1 = time.time()

        self.pseudotime = self.sess.run(self.tf_E_circular)
        self.rotation = self.sess.run(self.tf_V0)

        print('Full time %.2f' % (time.time() - t0))
        #saver.save(self.sess, './' + rec_file_name)

        return self.pseudotime, self.rotation


    def close(self):
        """
        Close the TensorFlow Session. All information about the model will be **deleted**.

        :return: None
        """
        self.sess.close()

    def save(self):
        """
        Save the Tensorflow Session for future use

        :return:
        """
        pass

    def generate(self, pseudotime=None):
        """
        Given a pseudo-time, generate a "ideal cycling cell"

        :type pseudotime: float or numpy.matrix
        :param pseudotime: the pseudo-time. If not specified it will use the whole inferred pseudotime.
        :return: the gene expression of a "ideal cycling cell"
        """
        if self.q_circular == 2:
            tf.concat([tf.cos(pseudotime), tf.sin(pseudotime)], 1) @ self.rotation
        else:
            np.concatenate([np.cos(pseudotime + i * 2 * np.pi / self.q_circular) for i in range(3)], axis=1) @ self.rotation

    def correct(self):
        """
        Correct the input expression matrix wrt cell cycle.

        :return:
        """
        return self.Y - self.generate()

    def predict(self):
        """
        Given a cell, predict its pseudo-time, given a fitted model.
        The model will not be fitted again.

        :return:
        """
        pass


class PreloadNaiveCyclum:
    def __init__(self, Y, q_circular=3, q_linear=0, seed=0):
        tf.set_random_seed(seed)
        self.q_circular = q_circular
        self.q_linear = q_linear
        self.q = q_linear + q_circular

        n, d = Y.shape

        self.tf_X = tf.constant(Y, dtype=tf.float32)

        # Linear Part
        self.tf_W = tf.Variable(tf.random_normal([d, self.q_linear]) / 4)
        self.tf_b = tf.Variable(tf.zeros([1, self.q_linear]))

        self.tf_E = (self.tf_X @ self.tf_W + self.tf_b)

        # Circular part
        self.tf_W_circular1 = tf.Variable(tf.random_normal([d, 30]))
        self.tf_b_circular1 = tf.Variable(tf.zeros([1, 30]))

        self.tf_E_circular1 = tf.tanh(self.tf_X @ self.tf_W_circular1 + self.tf_b_circular1)

        self.tf_W_circular2 = tf.Variable(tf.random_normal([30, 20]))
        self.tf_b_circular2 = tf.Variable(tf.zeros([1, 20]))

        self.tf_E_circular2 = tf.tanh(self.tf_E_circular1 @ self.tf_W_circular2 + self.tf_b_circular2)

        self.tf_W_circular3 = tf.Variable(tf.random_normal([20, 1]))
        self.tf_b_circular3 = tf.Variable(tf.zeros([1, 1]))

        self.tf_E_circular = self.tf_E_circular2 @ self.tf_W_circular3 + self.tf_b_circular3

        # slice
        self.tf_E_linear = self.tf_E

        tf_phase = lambda x, num: tf.concat([tf.cos(x + i * 2 * np.pi / num) for i in range(num)], 1)
        tf_cossin = lambda x: tf.concat([tf.cos(x), tf.sin(x)], 1)

        self.tf_Z_circular = tf_phase(self.tf_E_circular, 3)

        self.tf_Z_linear = self.tf_E_linear

        # decode
        self.tf_D_circular = self.tf_Z_circular
        self.tf_D_linear = self.tf_Z_linear

        self.tf_V0 = tf.Variable(tf.random_normal([self.q_circular, d]))
        self.tf_V1 = tf.Variable(tf.random_normal([self.q_linear, d]))

        self.tf_c = tf.Variable(tf.zeros([1, d]))

        self.tf_X_r_circular = self.tf_D_circular @ self.tf_V0
        self.tf_X_r = self.tf_X_r_circular + self.tf_D_linear @ self.tf_V1 + self.tf_c

        self.tf_R = tf.nn.l2_loss(self.tf_W) + tf.nn.l2_loss(self.tf_V1) + tf.nn.l2_loss(self.tf_V0)

        self.tf_L = tf.reduce_sum((self.tf_X - self.tf_X_r) ** 2)

    def fit(self):
        tf_burnin = tf.train.AdamOptimizer(50e-3).minimize(self.tf_L + self.tf_R * 50)
        tf_train = tf.train.AdamOptimizer(5e-3).minimize(self.tf_L + self.tf_R * 50)
        tf_refine = tf.train.AdamOptimizer(1e-3).minimize(self.tf_L + self.tf_R * 50)

        t1 = t0 = time.time()
        saver = tf.train.Saver()
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            sess.run(tf_burnin)
            if (i + 1) % 5000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(sess.run([self.tf_L, self.tf_R]), end="")
                print('time %.2f' % (time.time() - t1))
                t1 = time.time()

        for i in range(10000):
            sess.run(tf_train)
            if (i + 1) % 5000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(sess.run([self.tf_L, self.tf_R]), end="")
                print('time %.2f' % (time.time() - t1))
                t1 = time.time()

        for i in range(10000):
            sess.run(tf_refine)
            if (i + 1) % 5000 == 0:
                print('step %5d: loss ' % (i + 1), end="")
                print(sess.run([self.tf_L, self.tf_R]), end="")
                print('time %.2f' % (time.time() - t1))
                t1 = time.time()

        pseudotime = sess.run(self.tf_E_circular)
        rotation = sess.run(self.tf_V0)

        print(time.time() - t0)
        sess.close()

        return pseudotime, rotation


class Cyclum:
    """
    Wraps the mathematical method up and provide more functions.
    """