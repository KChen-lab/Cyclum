import numpy as np
import scipy as sp
import scipy.stats


def parzen_estimate(x, lim, half_granularity=100,
                    window=lambda x, rho: sp.stats.norm.pdf(x, 0, rho), scale=0.5):
    """Calculate parzen window estimation (a non-parametric density estimation method)

    :param x: instances
    :param lim: limit of domain
    :param half_granularity:
    :param window:
    :param scale:
    :return:
    """
    assert scale < 1 and scale > 0, "scale must be in (0, 1) to perform a unbiased estimation"
    gran = half_granularity * 2 + 1;

    n = len(x)
    rho = n ** -scale

    l = lim[1] - lim[0]
    grid = np.linspace(lim[0] - l / 2, lim[1] + l / 2, half_granularity * 4 + 1)
    discretized_window = window(grid, rho)
    discretized_window = discretized_window / np.sum(discretized_window)

    def individual(offset):
        indi = np.roll(discretized_window, offset - half_granularity)
        if offset > 0:
            indi[0: offset] = 0
        else:
            indi[offset:] = 0
        return indi

    xx = np.round((x - lim[0] + l) / l / 2 * gran)
    res = sum(map(individual, xx.astype(int).tolist())) / n
    return grid, res


def periodic_parzen_estimate(x, period=3.14, half_granularity=100,
                             window=lambda x, rho: sp.stats.norm.pdf(x, 0, rho), scale=0.5):
    """Calculate parzen window estimation specifically for periodic domain

    :param x:
    :param period:
    :param half_granularity:
    :param window:
    :param scale:
    :return:
    """
    assert scale < 1 and scale > 0, "scale must be in (0, 1) to perform a unbiased estimation"
    gran = half_granularity * 2 + 1

    n = len(x)
    rho = n ** -scale

    discretized_window = window(np.linspace(-period / 2, period / 2, gran), rho)
    discretized_window = discretized_window / np.sum(discretized_window)
    individual = lambda offset: np.roll(discretized_window, offset - half_granularity)
    xx = np.round((x % period) / period * gran) % gran
    res = sum(map(individual, xx.astype(int))) / n

    return np.linspace(0, period, gran), res


def precision_estimate(distr_vector_list, label_vector, possible_label_list):
    """Estimate precision

    :param distr_vector_list:
    :param label_vector:
    :param possible_label_list:
    :return:
    """
    distr_vector_list = list(map(np.squeeze, distr_vector_list))
    label_vector = np.reshape(label_vector, [-1, 1])
    n = np.sum(label_vector == np.array(possible_label_list), axis=0)
    n = n / np.sum(n)
    prob = np.vstack(distr_vector_list) * np.reshape(n, [-1, 1])
    return np.sum(np.max(prob, axis=0))

