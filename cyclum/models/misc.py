"""Implementation of cyclops in keras"""

from .ae import BaseAutoEncoder
import keras
import numpy
import sklearn.decomposition


class cyclops(BaseAutoEncoder):

    def __init__(self,
                 input_width: int,
                 linear_width: int = 0
                 ):
        super().__init__()

        y = keras.Input(shape=(input_width,), name='input')
        x00 = keras.layers.Dense(name='encoder_circular_in_0',
                                 units=1,
                                 kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                 bias_initializer=keras.initializers.Zeros()
                                 )(y)
        x01 = keras.layers.Dense(name='encoder_circular_in_1',
                                 units=1,
                                 kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                 bias_initializer=keras.initializers.Zeros()
                                 )(y)
        x002 = keras.layers.Lambda(keras.backend.square, name='x00_sqr')(x00)
        x012 = keras.layers.Lambda(keras.backend.square, name='x01_sqr')(x01)
        xx0 = keras.layers.Add(name='sqr_len')([x002, x012])
        xx0 = keras.layers.Lambda(keras.backend.sqrt, name='len')(xx0)
        x00 = keras.layers.Lambda(lambda x: x[0] / x[1], name='encoder_circular_out_0')([x00, xx0])
        x01 = keras.layers.Lambda(lambda x: x[0] / x[1], name='encoder_circular_out_1')([x01, xx0])

        if linear_width > 0:
            x1 = keras.layers.Dense(name='encoder_linear_out',
                                    units=linear_width,
                                    kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                    bias_initializer=keras.initializers.Zeros()
                                    )(y)

            x = keras.layers.Concatenate(name='embedding')([x00, x01, x1])
        else:
            x = keras.layers.Concatenate(name='embedding')([x00, x01])
        y_hat = keras.layers.Dense(name='output',
                                   units=input_width,
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                   bias_initializer=keras.initializers.Zeros()
                                   )(x)

        self.model = keras.Model(outputs=y_hat, inputs=y)

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

    def predict_pseudotime(self, data: numpy.ndarray):
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('embedding').output]
                                     )([data])
        return numpy.arctan2(res[0][:, 0], res[0][:, 1])

    @staticmethod
    def prep(data: numpy.ndarray, n_gene_kept: int = 10_000, variance_kept: float = 0.85) -> numpy.ndarray:
        mean = numpy.mean(data, axis=0)
        order = (-mean).argsort().argsort()
        data = data[:, order < n_gene_kept]  # order = 0 is the (0+1)st largest, = 9_999 is the (9_999 + 1)th largest
        data = numpy.clip(data, numpy.percentile(data, 2.5, axis=0), numpy.percentile(data, 97.5, axis=0))
        data = (data - numpy.mean(data, axis=0)) / numpy.mean(data, axis=0)
        pca_model = sklearn.decomposition.PCA()
        pca_embed = pca_model.fit_transform(data)
        cum_explained_variance = numpy.cumsum(pca_model.explained_variance_ratio_)
        print(cum_explained_variance)
        return pca_embed[:, 0:(sum(cum_explained_variance < variance_kept) + 1)]

