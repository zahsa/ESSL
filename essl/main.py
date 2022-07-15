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
import json


from essl.chromosome import chromosome_generator
from essl import fitness
from essl import mutate
from essl.crossover import PMX
from essl.crossover import onepoint_feas
from essl.utils import id_generator


@click.command()
@click.option("--pop_size", type=int, help="size of population")
@click.option("--num_generations",type=int, help="number of generations")
@click.option("--cxpb", default=0.2,type=float, help="probability of crossover")
@click.option("--mutpb", default=0.5,type=float, help="probability of mutation")
@click.option("--crossover", default="PMX",type=str, help="type of crossover (PMX, twopoint, onepoint)")
@click.option("--selection", default="SUS",type=str, help="type of selection (SUS, tournament)")
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
@click.option("--num_elite", default=0, type=int, help="number of elite chromosomes")
@click.option("--adaptive_pb", default=None, type=str, help="halving, generational")
@click.option("--patience", default=-1, type=int, help="number of non-improving generations before early stopping")
@click.option("--discrete_intensity", default=False, type=bool, help="whether or not to use discrete intensity vals")
@click.option("--use_test_acc", default=True, type=bool, help="whether or not to use test accuracy for fitness")

def main_cli(pop_size, num_generations,
                             cxpb,
                             mutpb,
                             crossover,
                             selection,
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
                             adaptive_pb,
                             patience,
                             discrete_intensity,
                             use_test_acc
                             ):
    main(pop_size=pop_size,
         num_generations=num_generations,
         cxpb=cxpb,
         mutpb=mutpb,
         crossover=crossover,
         selection=selection,
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
         adaptive_pb=adaptive_pb,
         patience=patience,
         discrete_intensity=discrete_intensity,
         use_test_acc=use_test_acc
         )


def main(pop_size, num_generations,
                             cxpb =  0.2,
                             mutpb = 0.5,
                             crossover = "PMX",
                             selection = "SUS",
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
                             num_elite=0,
                             adaptive_pb=None,
                             patience = -1,
                             discrete_intensity=False,
                             use_test_acc=True
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
    toolbox.register("gen_aug", chromosome_generator(length=chromosome_length,
                                                     discrete=discrete_intensity,
                                                     seed=seed))
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
                                     seed=seed,
                                     use_test_acc=use_test_acc)
    toolbox.register("evaluate", eval)
    if crossover == "PMX":
        toolbox.register("mate", PMX)
    elif crossover == "twopoint":
        toolbox.register("mate", tools.cxTwoPoint)
    elif crossover == "onepoint":
        toolbox.register("mate", tools.cxOnePoint)
    elif crossover == 'onepoint_feas':
        toolbox.register("mate", onepoint_feas)
    else:
        raise ValueError(f"invalid crossover ({crossover})")
    toolbox.register("mutate", mutate.mutGaussianChoice)
    if selection == "tournament":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif selection == "SUS":
        toolbox.register("select", tools.selStochasticUniversalSampling)
    elif selection == "roulette":
        toolbox.register("select", tools.selRoulette)

    # init pop and fitnesses #
    pop = toolbox.population(n=pop_size)
    # generate ids for each individual
    id_gen = id_generator()
    for ind in pop:
        ind.id = next(id_gen)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
       ind.fitness.values = fit
    outcomes = {m:[] for m in ["pop_vals", "min", "max", "avg", "std", "chromos"]}

    # Early stopping
    max_ind = pop[0]
    min_ind = pop[0]
    for ind in pop:
        if ind.fitness.values[0] < min_ind.fitness.values[0]:
            min_ind = ind
        elif ind.fitness.values[0] > max_ind.fitness.values[0]:
            max_ind = ind
    history = [max_ind.fitness.values[0]]
    no_improvement_count = 0

    mean = sum([f[0] for f in fitnesses]) / len(fitnesses)
    max_f = max_ind.fitness.values[0]

    # evolution loop
    for g in range(num_generations):
        print("-- Generation %i --" % g)
        # sort offspring in descending order
        elite_indexes = sorted(range(len(pop)), key=lambda i: pop[i].fitness.values[0], reverse=True)[:num_elite]
        elite = [pop[i] for i in elite_indexes]
        non_elite = [pop[i] for i in range(len(pop)) if i not in elite_indexes]

        # Select the next generation individuals
        offspring = toolbox.select(non_elite, len(non_elite))
        # Clone the selected individuals and elite individuals
        offspring = list(map(toolbox.clone, offspring)) + list(map(toolbox.clone, elite))
        random.shuffle(offspring)
        if adaptive_pb:
            if adaptive_pb == "halving":
                # drop_rate = 0.5, gen_drop = 3
                if not g % 3 and g != 0:
                    mutpb/=2
            elif adaptive_pb == "generational":
                cxpb = 1 - ((g + 1) / num_generations)
                mutpb = ((g + 1) / num_generations)
            elif adaptive_pb == "AGA":
                pass
            else:
                raise ValueError(f"invalid adaptive_pb value: {adaptive_pb}")
        # Apply crossover and mutation on the offspring
        # split list in two
        children = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if adaptive_pb == "AGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= mean:
                    cxpb = (max_f - f_p) / (max_f - mean)
                else:
                    cxpb = 1

            if random.random() < cxpb:

                toolbox.mate(child1, child2)
                # generate new ids for children
                child1.id = next(id_gen)
                child2.id = next(id_gen)
                children.append(child1.id)
                children.append(child2.id)
                del child1.fitness.values
                del child2.fitness.values


        for mutant in offspring:
            if adaptive_pb == "AGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= mean:
                    mutpb = (0.5 * (max_f - mutant.fitness.values[0])) / (max_f - mean)
                else:
                    mutpb = 0.5
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # generate new id for mutant
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

        min_ind = pop[0]
        max_ind = pop[0]
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
        outcomes["chromos"] += [[g, c] for c in pop]

        # early stopping
        if history[-1] > history[-2]:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        # stop if no improvement after two generations
        if no_improvement_count == patience:
            break
    if save_plots:
        plot_dir = os.path.join(exp_dir, "plots")
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        for m in outcomes:
            if m == "pop_vals":
                ax = sns.boxplot(data=pd.DataFrame(outcomes[m], columns=["gen", "fitness"]), x="gen", y="fitness", color='skyblue')
                ax.set_xlabel("Generation")
                ax.set_ylabel("Fitness")
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
    main(pop_size=2,
         ssl_epochs=1,
         num_generations=1,
         backbone="largerCNN_backbone",
         exp_dir="/home/noah/ESSL/exps/testing/adaptive",
         evaluate_downstream_kwargs={"num_epochs":1},
         crossover="onepoint",
         adaptive_pb="AGA"
         )
    print(f"TOOK {time.time()-t1} to run")

