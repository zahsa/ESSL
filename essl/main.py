import random
import click
from deap import base
from deap import creator
from deap import tools
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter
from datetime import datetime
import pandas as pd
sns.set_theme()
import copy
import json
from itertools import combinations
import multiprocessing


from essl.chromosome import chromosome_generator
from essl import fitness
from essl import mutate
from essl.crossover import PMX, cxOnePoint
from essl.utils import id_generator
from essl.parallel import dist_fit


@click.command()
@click.option("--pop_size", type=int, help="size of population")
@click.option("--num_generations",type=int, help="number of generations")
@click.option("--cxpb", default=0.2,type=float, help="probability of crossover")
@click.option("--mutpb", default=0.5,type=float, help="probability of mutation")
@click.option("--crossover", default="PMX",type=str, help="typer of crossover (PMX, twopoint)")
@click.option("--dataset", default="Cifar10",type=str, help="data set to use (Cifar10, )")
@click.option("--backbone", default="ResNet18_backbone",type=str, help="backbone to use (ResNet18_backbone, tinyCNN_backbone, largerCNN_backbone)")
@click.option("--ssl_task", default="SimCLR", type=str, help="SSL method (SimCLR)")
@click.option("--ssl_epochs", default=10, type=int, help="number of epochs for ssl task")
@click.option("--ssl_batch_size", default=256, type=int, help="batch size for ssl task")
@click.option("--evaluate_downstream_method", default="finetune", type=str, help="method of evaluation of ssl representation (finetune)")
@click.option("--device", default="cuda", type=str, help="device for torch (cuda, cpu)")
@click.option("--exp_dir", default="./", type=str, help="path to save experiment results")
@click.option("--use_tensorboard", default=True, type=bool, help="whether to use tensorboard or not")
@click.option("--save_plots", default=True, type=bool, help="whether to save plots or not")
@click.option("--chromosome_length", default=5, type=int, help="number of genes in chromosome")
@click.option("--num_elite", default=2, type=int, help="number of elite chromosomes")
@click.option("--adaptive_pbs", default=False, type=bool, help="whether to use adaptive mut and cx pb")
@click.option("--patience", default=3, type=int, help="number of non-improving generations before early stopping")
def main_cli(pop_size, num_generations,
                             cxpb,
                             mutpb,
                             crossover,
                             dataset,
                             backbone,
                             ssl_task,
                             ssl_epochs,
                             ssl_batch_size,
                             evaluate_downstream_method,
                             device,
                             exp_dir,
                             use_tensorboard,
                             save_plots,
                             chromosome_length,
                             num_elite,
                             adaptive_pbs,
                            patience):
    main(pop_size=pop_size,
         num_generations=num_generations,
         cxpb=cxpb,
         mutpb=mutpb,
         crossover=crossover,
         dataset=dataset,
         backbone=backbone,
         ssl_task=ssl_task,
         ssl_epochs=ssl_epochs,
         ssl_batch_size=ssl_batch_size,
         evaluate_downstream_method=evaluate_downstream_method,
         device=device,
         exp_dir=exp_dir,
         use_tensorboard=use_tensorboard,
         save_plots=save_plots,
         chromosome_length=chromosome_length,
         num_elite=num_elite,
         adaptive_pbs=adaptive_pbs,
         patience=patience)


def main(pop_size, num_generations,
                             cxpb =  0.2,
                             mutpb = 0.5,
                             crossover = "PMX",
                             dataset="Cifar10",
                             backbone="tinyCNN_backbone",
                             ssl_task="SimCLR",
                             ssl_epochs=1,
                             ssl_batch_size=256,
                             evaluate_downstream_method="finetune",
                             evaluate_downstream_kwargs={},
                             device="cuda",
                             exp_dir="./",
                             use_tensorboard=True,
                             save_plots=True,
                             chromosome_length=5,
                             seed=10,
                             num_elite=2,
                             adaptive_pbs=False,
                             num_workers=2,
                             patience = 3
                            ):

    # set seeds #
    random.seed(seed)

    if use_tensorboard:
        tb_dir = os.path.join(exp_dir, "tensorboard")
        if not os.path.isdir(tb_dir):
            os.mkdir(tb_dir)
        writer = SummaryWriter(os.path.join(tb_dir, datetime.now().strftime("%Y%m%d_%H:%M")))
    else:
        writer = None

    # init algo #
    toolbox = base.Toolbox()
    creator.create("Fitness", base.Fitness, weights=(100.0,)) # maximize accuracy
    creator.create("Individual", list, fitness=creator.Fitness, id=None)
    toolbox.register("gen_aug", chromosome_generator(length=chromosome_length, seed=seed))
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.gen_aug)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    eval = fitness.fitness_function(dataset=dataset,
                                     backbone=backbone,
                                     ssl_task=ssl_task,
                                     ssl_epochs=ssl_epochs,
                                     ssl_batch_size=ssl_batch_size,
                                     evaluate_downstream_method=evaluate_downstream_method,
                                     evaluate_downstream_kwargs=evaluate_downstream_kwargs,
                                     device=device,
                                     seed=seed)
    toolbox.register("evaluate", eval)
    if crossover == "PMX":
        toolbox.register("mate", PMX)
    elif crossover == "twopoint":
        toolbox.register("mate", tools.cxTwoPoint)
    elif crossover == "onepoint":
        toolbox.register("mate", cxOnePoint)
    else:
        raise ValueError(f"invalid crossover ({crossover})")
    toolbox.register("mutate", mutate.mutGaussian, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)


    # init pop and fitnesses #
    pop = toolbox.population(n=pop_size)
    # generate ids for each individual
    id_gen = id_generator()
    for ind in pop:
        ind.id = next(id_gen)

    # we want to parralelize this loop for N gpus
    # evaluate function takes a list of chromos and gpu number
    # outputs list of evaluations
    # def dist_fit(gpu, sub_pop):
    #     print(f"eval {len(sub_pop)} on gpu {gpu}")
    #     return gpu
    #
    chunk_size = int(len(pop)/num_workers)
    ranges = list(range(0, len(pop), chunk_size))
    chunks = []
    for i in range(len(ranges)-1):
        chunks.append([i, pop[ranges[i]:ranges[i+1]]])
    chunks.append([len(ranges)-1, pop[ranges[-1]:]])
    pool = multiprocessing.Pool(processes=num_workers)

    outputs = pool.starmap(dist_fit, [(gpu, sub_pop, toolbox.evaluate) for gpu, sub_pop in chunks])
    pool.close()
    import pdb;
    pdb.set_trace()
    # fitnesses = list(map(toolbox.evaluate, pop))

    # fitnesses = [[0] for i in range(len(pop))]
    for ind, fit in zip(pop, fitnesses):
       ind.fitness.values = fit
    outcomes = {m:[] for m in ["pop_vals", "min", "max", "avg", "std", "chromos"]}

    max_ind = pop[0].fitness.values[0]
    for ind in pop:
        if ind.fitness.values[0] > max_ind:
            max_ind = ind.fitness.values[0]
    history = [max_ind]
    no_improvement_count = 0
    # evolution loop
    for g in range(num_generations):
        print("-- Generation %i --" % g)
        if adaptive_pbs:
            cxpb = 1 - ((g+1) / num_generations)
            mutpb = ((g+1) / num_generations)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # sort offspring in descending order
        offspring.sort(key=lambda x: x.fitness.values[0], reverse=True)
        elite = offspring[:num_elite]
        non_elite = offspring[num_elite:]
        random.shuffle(non_elite)

        # Apply crossover and mutation on the offspring
        # split list in two
        for child1, child2 in zip(non_elite[::2], non_elite[1::2]): # Potenitally problematic, when uneven only does as many even pairs as possible
            if child1 != child2: # when using uneven numbers this method will combine the same chromo twice
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    # generate new ids for children
                    child1.id = next(id_gen)
                    child2.id = next(id_gen)
                    del child1.fitness.values
                    del child2.fitness.values


        for mutant in non_elite:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # generate new id for mutant
                mutant.id = next(id_gen)
                del mutant.fitness.values

        # combine non elite and elite
        offspring[:] = elite + non_elite

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
        history.append(max_f)
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
            # for now all losses are stored in memory, but this may cause memory problems later down
            # the line in. TO resolve this, we should remove chromosomes who are no longer in pop
            # (id not in pop)
            # clear dowsntream losses
            # toolbox.evaluate.downstream_losses = {k:v for k, v in toolbox.evaluate.downstream_losses.items() if k in }

        outcomes["min"].append(min_f)
        outcomes["max"].append(max_f)
        outcomes["avg"].append(mean)
        outcomes["std"].append(std)
        outcomes["pop_vals"]+=[[g, f] for f in fits]
        outcomes["chromos"]+=[[g, c] for c in pop]

        # early stopping
        if history[-1] > history[-2]:
            no_improvement_count = 0
        else:
            no_improvement_count+=1
        # stop if no improvement after two generations
        if no_improvement_count == patience:
            break

    if save_plots:
        plot_dir = os.path.join(exp_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        for m in outcomes:
            if m == "pop_vals":
                sns.boxplot(data=pd.DataFrame(outcomes[m], columns=["gen", "fitness"]), x="gen", y="fitness")
                plt.savefig(os.path.join(plot_dir, f"{m}.png"))
                plt.clf()
            elif m == "chromos":
                continue
            else:
                values = outcomes[m]
                sns.lineplot(list(range(len(values))), values)
                plt.savefig(os.path.join(plot_dir, f"{m}.png"))
                plt.clf()

    with open(os.path.join(exp_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes, f)


if __name__ == "__main__":
    import time
    t1 = time.time()
    main(pop_size=4,
         ssl_epochs=1,
         num_generations=2,
         backbone="tinyCNN_backbone",
         exp_dir="/home/noah/ESSL/experiments/iteration_7",
         evaluate_downstream_kwargs={"num_epochs":1},
         crossover="PMX"
         )
    print(f"TOOK {time.time()-t1} to run")