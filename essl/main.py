import random
from deap import base
from deap import creator
from deap import tools
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
from datetime import datetime
sns.set_theme()


from essl.chromosome import chromosome_generator
from essl import fitness
from essl import mutate
def main(pop_size, num_generations,
                             CXPB =  0.2,
                             MUTPB = 0.5,
                             dataset="Cifar10",
                             backbone="ResNet18_backbone",
                             ssl_task="SimCLR",
                             ssl_epochs=1,
                             ssl_batch_size=256,
                             evaluate_downstream_method="finetune",
                             device="cuda",
                             exp_dir="./",
                             use_tensorboard=True,
                             save_plots=True):

    if use_tensorboard:
        tb_dir = os.path.join(exp_dir, "tensorboard")
        if not os.path.isdir(tb_dir):
            os.mkdir(tb_dir)
        writer = SummaryWriter(os.path.join(tb_dir, datetime.now().strftime("%Y%m%d_%H:%M")))
    else:
        writer = None

    # init algo #
    toolbox = base.Toolbox()
    creator.create("Fitness", base.Fitness, weights=(-1.0,)) # minimize loss
    creator.create("Individual", list, fitness=creator.Fitness, id=None)
    toolbox.register("gen_aug", chromosome_generator())
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.gen_aug)
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
    def id():
        id = -1
        while True:
            id+=1
            yield id
    id_gen = id()
    for ind in pop:
        ind.id = next(id_gen)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    outcomes = {m:[] for m in ["pop_vals", "min", "max", "avg", "std"]}
    # evolution loop
    for g in range(num_generations):
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        # split list in two
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                child1.id = next(id_gen)
                child2.id = next(id_gen)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                mutant.id = next(id_gen)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        max_ind = pop[0]
        min_ind = pop[0]
        for ind in pop:
            if ind.fitness.values[0] < min_ind.fitness.values[0]:
                min_ind = ind
            elif ind.fitness.values[0] > max_ind.fitness.values[0]:
                max_ind = ind

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        min_f = min_ind.fitness.values[0]
        max_f = max_ind.fitness.values[0]

        print("  Min %s" % min_f)
        print("  Max %s" % max_f)
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        if writer:
            writer.add_scalar('Min Fitness', min_f, g)
            writer.add_scalar('Max Fitness', max_f, g)
            writer.add_scalar('Avg Fitness', mean, g)
            writer.add_scalar('Std dev', std, g)
            for f in fits:
                writer.add_scalar('population', f, g)
            # add losses for best and worst downstream eval
            for i, l in enumerate(toolbox.evaluate.downstream_losses[min_ind.id]):
                writer.add_scalar(f'worst downstream loss, gen {g}', l, i)
            writer.add_text(f"worst ind, gen {g}", str(min_ind))
            for i, l in enumerate(toolbox.evaluate.downstream_losses[max_ind.id]):
                writer.add_scalar(f'best downstream loss, gen {g}', l, i)
            writer.add_text(f"best ind, gen {g}", str(max_ind))
            # clear dowsntream losses
            toolbox.evaluate.downstream_losses = {}

        outcomes["min"].append(min_f)
        outcomes["max"].append(max_f)
        outcomes["avg"].append(mean)
        outcomes["std"].append(std)
        outcomes["pop_vals"].append(fits)

    if save_plots:
        plot_dir = os.path.join(exp_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        for m in outcomes:
            if m == "pop_vals":
                for g in outcomes[m]:
                    sns.lineplot(list(range(len(g))), g)
                plt.savefig(os.path.join(plot_dir, f"{m}.png"))
                plt.clf()
            else:
                values = outcomes[m]
                sns.lineplot(list(range(len(values))), values)
                plt.savefig(os.path.join(plot_dir, f"{m}.png"))
                plt.clf()


if __name__ == "__main__":
    import time
    t1 = time.time()
    main(pop_size=15,
         ssl_epochs=5,
         num_generations=20,
         backbone="tinyCNN_backbone",
         exp_dir="/home/noah/ESSL/experiments/test_1")
    print(f"TOOK {time.time()-t1} to run")