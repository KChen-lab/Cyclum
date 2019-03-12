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

        :return: pseudotime and rotation matrix
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

        pseudotime = self.sess.run(self.tf_E_circular)
        rotation = self.sess.run(self.tf_V0)

        print('Full time %.2f' % (time.time() - t0))
        #saver.save(self.sess, './' + rec_file_name)

        return pseudotime, rotation


    def close(self):
        """
        Close the TensorFlow Session. The model will be **deleted**.

        :return: None
        """
        self.sess.close()

    def save(self):
        """
        Save the Tensorflow Session for future use

        :return:
        """
        pass

    def generate(self):
        """
        Given a pseudo-time, generate a "standard" cell

        :return:
        """
        pass

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