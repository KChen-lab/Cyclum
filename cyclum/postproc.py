import numpy as np


def linear_divide(x, a, b, c):
    """
    Find the best two dividing point for a linear array made up with three different characters.
    """
    n = len(x)
    ab_penalty = (np.append(0, (np.array(x) == b).cumsum()) +
                  np.append((np.array(x) == a)[::-1].cumsum()[::-1], 0))
    bc_penalty = (np.append(0, (np.array(x) == c).cumsum()) +
                  np.append((np.array(x) == b)[::-1].cumsum()[::-1], 0))
    return (ab_penalty.argmin(), n - bc_penalty[::-1].argmin()), ab_penalty.min() + bc_penalty.min()


def circular_divide(x, a, b, c):
    """
    Find the best three dividing point for a circular array made up with three different characters.
    """
    n = len(x)
    best_loc = None
    best_penalty = float("inf")
    for i in range(n):
        loc, penalty = linear_divide(np.append(x[i:], x[:i]), a, b, c)
        if loc[1] > loc[0] and penalty < best_penalty:
            best_penalty = penalty
            best_loc = (i, (loc[0] + i) % n, (loc[1] + i) % n)
    return best_loc, best_penalty


def circular_divide2(x, a, b, c):
    """
        Find the best three dividing point for a circular array made up with three different characters,
        in either direction.
    """
    n = len(x)
    best_loc, best_penalty = circular_divide(x, a, b, c)
    reverse_best_loc, reverse_best_penalty = circular_divide(x[::-1], a, b, c)

    res = np.array([' ' * max(len(a), len(b), len(c))] * n)

    def wrap_around_assign(arr, i, j, v):
        if j >= i:
            arr[i:j] = v
        else:
            arr[i:] = v
            arr[:j] = v

    if best_penalty <= reverse_best_penalty:
        values_assigned = (a, b, c)
    else:
        best_loc = [n - i for i in reverse_best_loc[::-1]]
        best_penalty = reverse_best_penalty
        values_assigned = (b, a, c)
    for i in range(3):
        wrap_around_assign(res, best_loc[i], best_loc[(i + 1) % 3], values_assigned[i])
    return res

def refine_labels(pseudotime, original_labels):
    """
    Refine labels using pseudotime.
    """
    if len(pseudotime) != len(original_labels):
        raise ValueError("Lengths of pseudotime and labels must be the same.")
    unique_labels = sorted(np.unique(original_labels))
    if len(unique_labels) != 3:
        raise ValueError("Only supports 3 classes.")
    order = np.argsort(pseudotime)
    refined_labels = np.array([' ' * max(len(i) for i in unique_labels)] * len(pseudotime))
    refined_labels[order] = circular_divide2(np.array(original_labels)[order], *unique_labels)
    return refined_labels