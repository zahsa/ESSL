import random
from deap import base
from deap import creator
from deap import tools

from essl.chromosome import chromosome_generator
from essl import fitness
from essl import mutate
def main(pop_size, num_generations,
                     dataset="Cifar10",
                     backbone="ResNet18_backbone",
                     ssl_task="SimCLR",
                     ssl_epochs=1,
                     ssl_batch_size=256,
                     evaluate_downstream_method="finetune",
                     device="cuda"):
    # init algo #
    toolbox = base.Toolbox()
    toolbox.register("individual", chromosome_generator())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    eval = fitness.fitness_function(dataset=dataset,
                                     backbone=backbone,
                                     ssl_task=ssl_task,
                                     ssl_epochs=ssl_epochs,
                                     ssl_batch_size=ssl_batch_size,
                                     evaluate_downstream_method=evaluate_downstream_method,
                                     device=device)
    toolbox.register("evaluate", eval)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate.mutGaussian, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # init pop and fitnesses #
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2
    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]
    for g in num_generations:
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)





if __name__ == "__main__":
    main(pop_size=5, num_generations=3)