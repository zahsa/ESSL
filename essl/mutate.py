"""
perturbation of intensities
    - must remain within the feasible space
    - pre or post
"""


from essl.ops import DEFAULT_OPS
import math
import random

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

class MutateOperators:

    def __init__(self,chromosome,fitness,avg_fitness,mute_rate,crossover_rate):
        """
        """
        self.chromosome = chromosome
        self.fitness = fitness
        self.avg_fitness = avg_fitness
        self.mute_rate = mute_rate
        self.crossover_rate = crossover_rate
        self.parents = parents

    def mutateOp1(self):
        if self.chrom_fitness < self.avg_fitness:
            mute_rate *= 2
        else:
            mute_rate /= 2

        if np.random.rand() <= mute_rate:
            noise = np.random.uniform(0, 1, 1)
            self.chromosome['intensity'] += noise



def mutGaussian(individual,  mu=0, sigma=1, indpb=0.05):
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
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

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






