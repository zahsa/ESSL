from essl.ops import DEFAULT_OPS
import random
import numpy as np

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

def mutGaussian(individual, indpb=0.05, discrete=False, intensity_increments=10):
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
    if discrete:
        for i in range(1, size):
            if random.random() < indpb:
                i_range = DEFAULT_OPS[individual[i][0]]
                increment = abs(i_range[1] - i_range[0]) / intensity_increments
                if isinstance(i_range[0], int):
                    update =  int(individual[i][1] + (random.choice([-1, 1]) * increment))
                    if update < i_range[0]:
                        update = int(individual[i][1] + abs(increment))
                    elif update > i_range[1]:
                        update = int(individual[i][1] - increment)
                else:
                    # update = float(round(random.choice(np.arange(*i_range, increment, dtype=float)), 2))
                    update = float(round((random.choice([-1, 1]) * increment) + individual[i][1], 2))
                    if update < i_range[0]:
                        update = float(round(individual[i][1] + abs(increment), 2))
                    elif update > i_range[1]:
                        update = float(round(individual[i][1] - increment, 2))
                # if update <= DEFAULT_OPS[individual[i][0]][1] and update >= DEFAULT_OPS[individual[i][0]][0]:
                individual[i][1] = update
    else:
        for i in range(1, size):
            if random.random() < indpb:
                i_range = DEFAULT_OPS[individual[i][0]]
                if isinstance(i_range[0], int):
                    update = (random.choice([-1, 1]) + individual[i][1])
                    # max out range
                    if update < DEFAULT_OPS[individual[i][0]][0]:
                        update = individual[i][1] + 1
                    elif update > DEFAULT_OPS[individual[i][0]][1]:
                        update = individual[i][1] - 1
                else:
                    m = abs(i_range[1] - i_range[0]) / 4
                    s = m / 2
                    sign = random.choice([1, -1])
                    gauss = random.gauss(m, s)*sign
                    update = individual[i][1] + gauss
                    # max out range
                    if update < DEFAULT_OPS[individual[i][0]][0]:
                        update = individual[i][1] + abs(gauss)
                    elif update > DEFAULT_OPS[individual[i][0]][1]:
                        update = individual[i][1] - gauss
                individual[i][1] = update
    return individual,



def mutGaussianChoice(individual,  discrete=False, intensity_increments=10):
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
    num_genes = random.choice(range(1, size-1))
    genes = random.sample(range(1, size), num_genes)
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
                # max out range
                if update < DEFAULT_OPS[individual[i][0]][0]:
                    update = individual[i][1] + 1
                elif update > DEFAULT_OPS[individual[i][0]][1]:
                    update = individual[i][1] - 1
            else:
                m = abs(i_range[1] - i_range[0]) / 4
                s = m / 2
                sign = random.choice([1, -1])
                gauss = random.gauss(m, s) * sign
                update = individual[i][1] + gauss
                # max out range
                if update < DEFAULT_OPS[individual[i][0]][0]:
                    update = individual[i][1] + abs(gauss)
                elif update > DEFAULT_OPS[individual[i][0]][1]:
                    update = individual[i][1] - gauss
            individual[i][1] = update
    return individual,


if __name__ == "__main__":
    from essl.chromosome import chromosome_generator
    cc = chromosome_generator()
    x = cc()
    for _ in range(10):
        print(mutGaussianChoice(x))





