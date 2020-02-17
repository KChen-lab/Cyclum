"""Main module."""

from math import pi
import keras
import time
import numpy
from typing import List, Union, Callable
import sklearn.decomposition
import sklearn.metrics


class BaseAutoEncoder(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def circular_unit(name: str, comp: int = 2) -> Callable:
        """Create a circular unit

        :param name: Name of this unit
        :param comp: components of phases. 2
        :return: function f: input tensor -> output tensor
        """
        def func(x):
            out = []
            if comp < 2:
                raise ValueError("comp must be at least 2")
            elif comp == 2:
                out = [keras.layers.Lambda(lambda x: keras.backend.sin(x), name=name + '_sin')(x),
                       keras.layers.Lambda(lambda x: keras.backend.cos(x), name=name + '_cos')(x)]
            else:
                out = [
                    keras.layers.Lambda(lambda x: keras.backend.sin(x + 2 * pi * i / comp), name=name + '_' + str(i))(x)
                    for i in range(comp)]
            out = keras.layers.Concatenate(name=name + '_out')(out)
            return out

        return func

    @staticmethod
    def logistic_unit(name: str, n: int, trans: bool = True, reg_scale: float = 1e-2, reg_trans: float = 1e-2) -> Callable:
        """Create a logistic unit

        :param name: Name of this unit
        :param n: Number of perceptrons
        :param trans: Allow translation (i.e. b in Ax + b)
        :param reg_scale: regularization on scaling (i.e. A in Ax + b)
        :param reg_trans: regularization of translation
        :return: function f: input tensor -> output tensor
        """
        def func(x):
            x = keras.layers.Dense(name=name + '_scale',
                                   units=n,
                                   use_bias=trans,
                                   kernel_regularizer=keras.regularizers.l2(reg_scale),
                                   bias_regularizer=keras.regularizers.l2(reg_trans),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                   bias_initializer=keras.initializers.Zeros()
                                   )(x)
            x = keras.layers.Activation(name=name + '_out',
                                        activation='tanh'
                                        )(x)
            return x

        return func

    @staticmethod
    def linear_unit(name: str, n: int, trans: bool = True, reg_scale: float = 1e-2, reg_trans: float = 1e-2) -> Callable:
        """Create a logistic unit

        :param name: Name of this unit
        :param n: Number of perceptrons
        :param trans: Allow translation (i.e. b in Ax + b)
        :param reg_scale: regularization on scaling (i.e. A in Ax + b)
        :param reg_trans: regularization of translation
        :return: function f: input tensor -> output tensor
        """
        def func(x):
            x = keras.layers.Dense(name=name + '_scale',
                                   units=n,
                                   use_bias=trans,
                                   kernel_regularizer=keras.regularizers.l2(reg_scale),
                                   bias_regularizer=keras.regularizers.l2(reg_trans),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                   bias_initializer=keras.initializers.Zeros()
                                   )(x)
            return x

        return func

    @staticmethod
    def encoder(name: str, size: List[int], reg: float, drop: float, act: Union[str, Callable] = 'tanh') -> Callable:
        """Create a nonlinear encoder

        :param name: Name of this unit
        :param size: Size of each layer
        :param reg: regularization strength
        :param drop: dropout rate
        :param act: activation function
        :return: function f: input tensor -> output tensor
        """
        def func(x):
            for i, w in enumerate(size):
                x = keras.layers.Dense(name=name + str(i) + '_scale',
                                       units=w,
                                       kernel_regularizer=keras.regularizers.l2(reg),
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                       )(x)
                if drop > 0:
                    x = keras.layers.Dropout(name=name + str(i) + '_dropout',
                                             rate=drop
                                             )(x)
                x = keras.layers.Activation(name=name + str(i) + '_act',
                                            activation=act
                                            )(x)

            x = keras.layers.Dense(name=name + '_out',
                                   units=1,
                                   use_bias=False,
                                   #kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None)
                                   )(x)
            return x

        return func

    @staticmethod
    def linear_bypass(name: str, n: int, reg: float) -> Callable:
        """Create a linear encoder

        :param name:
        :param n:
        :param reg:
        :return: function f: input tensor -> output tensor
        """
        def func(x):
            x = keras.layers.Dense(name=name + '_out',
                                   units=n,
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None)
                                   )(x)
            return x

        return func

    @staticmethod
    def decoder(name: str, n: int) -> Callable:
        """Create a dncoder

        :param name:
        :param n: Output width
        :return: function f: input tensor -> output tensor
        """
        def func(x: list):
            if len(x) > 1:
                x = keras.layers.Concatenate(name=name + '_concat')(x)
            else:
                x = x[0]
            x = keras.layers.Dense(name=name + '_out',
                                   units=n,
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.Zeros()
                                   )(x)
            return x

        return func

    def save(self, filepath):
        """Save a BaseAutoEncoder object

        :param filepath: h5 suffix is recommended, i.e., filename.h5
        :return:
        """
        self.model.save(filepath)

    def load(self, filepath):
        """Load a BaseAutoEncoder object

        :param filepath:
        :return:
        """
        self.model = keras.models.load_model(filepath, custom_objects={"keras": keras, "keras.backend": keras.backend})

    class MyCallback(keras.callbacks.Callback):
        """Call back function for

        :param interval: report loss, time, etc. per interval epochs
        """
        def __init__(self, interval):
            super().__init__()
            self.cnt = 0
            self.interval = interval
            self.start_time = 0
            self.rec = {'time': [], 'loss': []}

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, batch, logs=None):
            self.cnt += 1
            self.rec['time'].append(time.time() - self.start_time)
            self.rec['loss'].append(logs.get('loss'))
            if self.cnt % self.interval == 0:
                print(f'epoch: {self.cnt}/{self.params["epochs"]}, loss: {logs.get("loss") : .4f}, '
                      f'time elapsed: {self.rec["time"][-1] : .2f}s, '
                      f'time left: {((self.params["epochs"] / self.cnt - 1) * self.rec["time"][-1]) : .2f}s')

    def show_structure(self):
        """Show the structure of the network

        :return: The graph for the structure
        """
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        return SVG(model_to_dot(self.model, show_shapes=True).create(prog='dot', format='svg'))

class AutoEncoder(BaseAutoEncoder):
    """A Cyclum style autoencoder

    :param input_width: width of input, i.e., number of genes
    :param encoder_depth: depth of encoder, i.e., number of *hidden* layers
    :param encoder_width: width of encoder, one of the following:

        * An integer stands for number of nodes per layer. All hidden layers will have the same number of nodes.
        * A list, whose length is equal to `encoder_depth`, of integers stand for numbers of nodes of the layers.

    :param n_circular_unit: 0 or 1, number of circular unit; may add support for 1+ in the future.
    :param n_logistic_unit: number of logistic (tanh) unit which *runs on the circular embedding*. Under testing.
    :param n_linear_unit: number of linear unit which runs *on the circular embedding*. Under testing.
    :param n_linear_bypass: number of linear components.
    :param dropout_rate: rate for dropout.
    :param nonlinear_reg: strength of regularization on the nonlinear encoder.
    :param linear_reg: strength of regularization on the linear encoder.
    :param filepath: filepath of stored model. If specified, all other parameters are omitted.
    """
    def __init__(self,
                 input_width: int = None,
                 encoder_depth: int = 2,
                 encoder_width: Union[int, List[int]] = 50,
                 n_circular_unit: int = 1,
                 n_logistic_unit: int = 0,
                 n_linear_unit: int = 0,
                 n_linear_bypass: int = 0,
                 dropout_rate: float = 0.0,
                 nonlinear_reg: float = 1e-4,
                 linear_reg: float = 1e-4,
                 filepath: str = None
                 ):

        super().__init__()

        if input_width is None:
            self.load(filepath)
        else:
            if type(encoder_width) is int:
                encoder_size = [encoder_width] * encoder_depth
            elif type(encoder_width) is list and len(encoder_width) == encoder_depth:
                encoder_size = encoder_width
            else:
                raise ValueError("encoder_width must be either (1) an integer or (2) a list of integer, whose length is "
                                 "equal to encoder_depth.")

            y = keras.Input(shape=(input_width,), name='input')
            x = self.encoder('encoder', encoder_size, nonlinear_reg, dropout_rate, 'tanh')(y)

            chest = []
            if n_linear_bypass > 0:
                x_bypass = self.linear_bypass('bypass', n_linear_bypass, linear_reg)(y)
                chest.append(x_bypass)
            if n_logistic_unit > 0:
                x_logistic = self.logistic_unit('logistic', n_logistic_unit)(x)
                chest.append(x_logistic)
            if n_linear_unit > 0:
                x_linear = self.linear_unit('linear', n_linear_unit)(x)
                chest.append(x_linear)
            if n_circular_unit > 0:
                x_circular = self.circular_unit('circular')(x)
                chest.append(x_circular)
            y_hat = self.decoder('decoder', input_width)(chest)

            self.model = keras.Model(outputs=y_hat, inputs=y)

    def pre_train(self, data, n_linear_bypass: int, epochs: int = 100, verbose: int = 10, rate: float = 1e-4):
        """Train the network with PCA. May save some training time. Only applicable to circular with linear bypass.

        :param data: data used
        :param n_linear_bypass: number of linear bypasses, must be the same as the one specified during the init.
        :param epochs: training epochs
        :param verbose: per how many epochs does it report the loss, time consumption, etc.
        :param rate: training rate
        :return: history of loss
        """
        pca_model = sklearn.decomposition.PCA(n_components=n_linear_bypass + 2)
        pca_load = pca_model.fit_transform(data)

        stdd_pca_load = pca_load / numpy.std(pca_load, axis=0) / 3

        if n_linear_bypass > 0:
            pretrain_model = keras.Model(outputs=self.model.get_layer('decoder_concat').output,
                                         inputs=self.model.get_layer('input').input)
        else:
            pretrain_model = keras.Model(outputs=self.model.get_layer('circular_out').output,
                                         inputs=self.model.get_layer('input').input)

        my_callback = self.MyCallback(verbose)
        pretrain_model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(rate))
        history = pretrain_model.fit(data, stdd_pca_load, epochs=epochs, verbose=0, callbacks=[my_callback])

        return history

    def train(self, data, batch_size: int = None, epochs: int = 100, verbose: int = 10, rate: float = 1e-4):
        """Train the model. It will not reset the weights each time so it can be called iteratively.

        :param data: data used for training
        :param batch_size: batch size for training, if unspecified default to 32 as is set by keras
        :param epochs: number of epochs in training
        :param verbose: per how many epochs does it report the loss, time consumption, etc.
        :param rate: training rate
        :return: history of loss
        """
        self.model.compile(loss='mean_squared_error',
                           optimizer=keras.optimizers.Adam(rate))

        my_callback = self.MyCallback(verbose)
        history = self.model.fit(data, data, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[my_callback])

        return history

    def predict_pseudotime(self, data):
        """Predict the circular pseudotime

        :param data: data to be used for training
        :return: the circular pseudotime
        """
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('encoder_out').output]
                                     )([data])
        return res[0]

    def predict_linear_bypass(self, data):
        """Predict the linear bypass loadings.

        :param data: data to be used for training
        :return: the circular pseudotime
        """
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('bypass_out').output]
                                     )([data])
        return res[0]

    def get_weight(self):
        """Get the weight of the transform, where the last two dimensions are for the sinusoidal unit

        :return: a matrix
        """
        layer = self.model.get_layer('decoder_out')
        return layer.get_weights()[0]
