"""Auto tuning."""

import cyclum.models

import sklearn.decomposition
import sklearn.metrics
import math


class CyclumAutoTune(cyclum.models.AutoEncoder):
    """Circular autoencoder with automatically decided number of linear components

        We first perform PCA on the data, and record the MSE of having first 1, 2, ..., max_linear_dims + 1 components.
        We then try to train a circular autoencoder with 0, 1, ..., max_linear_dims linear components.
        We compare circular autoencoder with i linear components with PCA with (i + 1) components, for i = 0, 1, ...
        We record the first i where the difference of loss compared with PCA is greater than both (i - 1) and (i + 1),
        or just (i + 1) if i == 0.

        At the end, this class will be a UNTRAINED model, which has optimal numbers of linear components.
        You can train it will all your data, more epochs, and better learning rate.

        :param data: The data used to decide number of linear components. For a large dataset, you may use a representative portion of it.
        :param max_linear_dims: maximum number of linear dimensions.
        :param epochs: number of epochs for each test
        :param verbose: per how many epochs does it report the loss, time consumption, etc.
        :param rate: training rate
        :param early_stop: Stop checking more linear components when result decided? ONLY affects the elbow plot. NO influence on result.
        :param encoder_depth: depth of encoder, i.e., number of *hidden* layers
        :param encoder_width: width of encoder, one of the following:

            * An integer stands for number of nodes per layer. All hidden layers will have the same number of nodes.
            * A list, whose length is equal to `encoder_depth`, of integers stand for numbers of nodes of the layers.

        :param dropout_rate: rate for dropout.
        :param nonlinear_reg: strength of regularization on the nonlinear encoder.
        :param linear_reg: strength of regularization on the linear encoder.

        Examples:
            >>> from cyclum.hdfrw import hdf2mat, mat2hdf
            >>> df = hdf2mat('path_to_hdf.h5')
            >>> m = CyclumAutoTune(df.values, max_linear_dims=5)
            >>> m.train(df.values)
            >>> pseudotime = m.predict_pseudotime(df.values)
            >>> mat2hdf(pseudotime, 'path_to_pseudotime.h5')
    """
    def __init__(self, data, max_linear_dims=3, epochs=500, verbose=100, rate=5e-4, early_stop=False,
                 encoder_depth=2, encoder_width=50, dropout_rate=0.1, nonlinear_reg=1e-4, linear_reg=1e-4):
        print("Auto tuning number of linear components...")

        self.max_linear_dims = max_linear_dims

        print("Performing PCA...")
        pca_model = sklearn.decomposition.PCA(n_components=self.max_linear_dims + 2)
        pca_load = pca_model.fit_transform(data)
        pca_comp = pca_model.components_
        self.pca_loss = [sklearn.metrics.mean_squared_error(data, pca_load[:, 0:(i + 1)] @ pca_comp[0:(i + 1), :]) for i
                         in range(self.max_linear_dims + 2)]

        print("Training Autoencoder with...")
        self.ae_loss = []
        best_n_linear_dims = None
        for i in range(self.max_linear_dims + 1):
            print(f"    {i} linear dimensions...")
            model = cyclum.models.AutoEncoder(input_width=data.shape[1],
                                              encoder_depth=encoder_depth,
                                              encoder_width=encoder_width,
                                              n_circular_unit=1,
                                              n_logistic_unit=0,
                                              n_linear_unit=0,
                                              n_linear_bypass=i,
                                              dropout_rate=dropout_rate,
                                              nonlinear_reg=nonlinear_reg,
                                              linear_reg=linear_reg)
            history = model.train(data, epochs=epochs, verbose=verbose, rate=rate)

            self.ae_loss.append(history.history['loss'][-1])

            print(self.ae_loss)
            print(self.pca_loss)

            if i > 0 and best_n_linear_dims is None and ((self.pca_loss[i] - self.ae_loss[i]) / self.pca_loss[i] <
                (self.pca_loss[i - 1] - self.ae_loss[i - 1]) / self.pca_loss[i - 1]):
                best_n_linear_dims = i - 1
                print(f"Found! Use {best_n_linear_dims} linear components...")
                if early_stop:
                    break
                else:
                    print("Early stop disabled, continue to check all cases...")

        if best_n_linear_dims is None:
            best_n_linear_dims = max_linear_dims
            print(f"Have not found one. Suggest raise max_linear_dims. Use {max_linear_dims} linear components...")

        super().__init__(input_width=data.shape[1],
                         encoder_depth=encoder_depth,
                         encoder_width=encoder_width,
                         n_circular_unit=1,
                         n_logistic_unit=0,
                         n_linear_unit=0,
                         n_linear_bypass=best_n_linear_dims,
                         dropout_rate=dropout_rate,
                         nonlinear_reg=nonlinear_reg,
                         linear_reg=linear_reg)

    def show_elbow(self):
        """Show an elbow plot of both PCA and autoencoder
        You will observe the time when autoencoder become to have a higher loss than PCA.
        The previous time is considered as the best model.

        :return: figure object
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(list(range(len(self.pca_loss))), self.pca_loss, "+-")
        plt.plot(list(range(len(self.ae_loss))), self.ae_loss, "x-")
        plt.legend(['PCA', 'AE'])
        plt.xticks(range(max(map(len, [self.pca_loss, self.ae_loss]))))
        plt.xlabel('X: X linear components or (X+1) PCs')
        plt.ylabel('Mean squared error')

        return fig

    def show_bar(self, root=False):
        """Show a bar plot for what percentage of more loss is handled by the circular component

        :return: figure object
        """
        import matplotlib.pyplot as plt
        import numpy
        fig = plt.figure()
        if root:
            linear_handled = numpy.sqrt(self.pca_loss[:-1])# - numpy.sqrt(self.pca_loss[1:])
            circular_handled = numpy.sqrt(self.pca_loss[:-1]) - numpy.sqrt(self.ae_loss)
        else:
            linear_handled = numpy.array(self.pca_loss[:-1])# - numpy.array(self.pca_loss[1:])
            circular_handled = numpy.array(self.pca_loss[:-1]) - numpy.array(self.ae_loss)
        plt.bar(list(range(len(self.ae_loss))), circular_handled / linear_handled)
        plt.xticks(list(range(len(self.ae_loss))))
        plt.xlabel('Dimensionality')
        plt.ylabel('Relative decrease in MSE')

        return fig