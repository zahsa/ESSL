import random
from deap import base
from deap import creator
from deap import tools

from essl.chromosome import chromosome_generator
import fitness

def main(pop_size, fitness_function, dataset, ):
    # init algo #
    toolbox = base.Toolbox()
    toolbox.register("individual", chromosome_generator())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.population(n=pop_size)

    # get eval function #
    eval = fitness.__dict__[fitness_function]

    toolbox.register("evaluate", eval)


if __name__ == "__main__":
    main(300, fitness_function="eval_chromosome")