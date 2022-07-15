from essl.ops import DEFAULT_OPS
import random
import numpy as np

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

def mutGaussian(individual,  mu=0, sigma=1, indpb=0.05, seed=10, discrete=False, intensity_increments=10):
    """
    DIrectly modified from source code to work with our chromosomes

    taps out with default ops preset ranges

    This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    random.seed(seed)
    size = len(individual)
    import pdb;pdb.set_trace()
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    if discrete:
        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                i_range = DEFAULT_OPS[individual[i][0]]
                increment = abs(i_range[1] - i_range[0]) / intensity_increments
                if isinstance(i_range[0], int):
                    # update = int(random.choice(np.arange(*i_range, increment, dtype=int)))
                    update =  int((random.choice([-1, 1]) * increment) + individual[i][1])
                else:
                    # update = float(round(random.choice(np.arange(*i_range, increment, dtype=float)), 2))
                    update = float(round((random.choice([-1, 1]) * increment) + individual[i][1], 2))
                if update <= DEFAULT_OPS[individual[i][0]][1] and update >= DEFAULT_OPS[individual[i][0]][0]:
                    individual[i][1] = update
    else:
        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                update = individual[i][1] + random.gauss(m, s)
                # max out range
                if update < DEFAULT_OPS[individual[i][0]][0]:
                    update = DEFAULT_OPS[individual[i][0]][0]
                elif update > DEFAULT_OPS[individual[i][0]][1]:
                    update = DEFAULT_OPS[individual[i][0]][1]
                individual[i][1] = update
    return individual,



def mutGaussianChoice(individual,  mu=0, sigma=1, discrete=False, intensity_increments=10):
    """
    Randomly selects 1-N genes to mutate and applies mutation
    if discrete, increments in either negative or positive direction by increment size

    if discrete value
    """
    size = len(individual)
    num_genes = random.choice(range(1, size + 1))
    genes = random.sample(range(size), num_genes)
    if discrete:
        for i in genes:
            i_range = DEFAULT_OPS[individual[i][0]]
            increment = abs(i_range[1] - i_range[0]) / intensity_increments
            if isinstance(i_range[0], int):
                update = int((random.choice([-1, 1]) * increment) + individual[i][1])
            else:
                update = float(round((random.choice([-1, 1]) * increment) + individual[i][1], 2))
            if update <= DEFAULT_OPS[individual[i][0]][1] and update >= DEFAULT_OPS[individual[i][0]][0]:
                individual[i][1] = update
    else:
        for i in genes:
            i_range = DEFAULT_OPS[individual[i][0]]
            if isinstance(i_range[0], int):
                update = (random.choice([-1, 1]) + individual[i][1])
            else:
                update = individual[i][1] + random.gauss(mu, sigma)
            # max out range
            if update < DEFAULT_OPS[individual[i][0]][0]:
                update = DEFAULT_OPS[individual[i][0]][0]
            elif update > DEFAULT_OPS[individual[i][0]][1]:
                update = DEFAULT_OPS[individual[i][0]][1]
            individual[i][1] = update

    return individual,








