import random
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
import numpy as np


from essl.chromosome import chromosome_generator, chromosome_generator_mo, SSL_TASKS
from essl import fitness
from essl import mutate
from essl.crossover import PMX, PMX_mo
from essl.crossover import onepoint_feas, onepoint_feas_mo
from essl.utils import id_generator


def GA(pop_size, num_generations,
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
                             eval_method="final test"
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
                                                     discrete=discrete_intensity
                                                     ))
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
                                     eval_method=eval_method)
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

    f_bar = sum([f[0] for f in fitnesses]) / len(fitnesses)
    averages = [f_bar]
    f_global_bar = sum(averages)/len(averages)
    global_max_mean = f_bar
    max_f = max_ind.fitness.values[0]
    global_max = max_f
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
        offspring = list(map(toolbox.clone, offspring))
        random.shuffle(offspring)
        if adaptive_pb:
            if adaptive_pb == "halving":
                # drop_rate = 0.5, gen_drop = 3
                if not g % 3 and g != 0:
                    mutpb/=2
            elif adaptive_pb == "generational":
                cxpb = 1 - ((g + 1) / num_generations)
                mutpb = ((g + 1) / num_generations)
            elif adaptive_pb in ["AGA", "GAGA", "GAGA_V2"]:
                pass
            else:
                raise ValueError(f"invalid adaptive_pb value: {adaptive_pb}")
        # Apply crossover and mutation on the offspring
        # split list in two
        children = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if adaptive_pb == "AGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb = (max_f - f_p) / (max_f - f_bar)
                else:
                    cxpb = 1
            elif adaptive_pb == "GAGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= global_max_mean:
                    cxpb = (global_max - f_p) / (global_max - global_max_mean)
                else:
                    cxpb = 1
            elif adaptive_pb == "GAGA_V2":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb = (global_max - f_p) / (global_max - f_global_bar)
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
                if mutant.fitness.values[0] >= f_bar:
                    mutpb = (0.5 * (max_f - mutant.fitness.values[0])) / (max_f - f_bar)
                else:
                    mutpb = 0.5
            elif adaptive_pb == "GAGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= global_max_mean:
                    mutpb = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - global_max_mean)
                else:
                    mutpb = 0.5

            elif adaptive_pb == "GAGA_V2":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= f_bar:
                    mutpb = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - f_global_bar)
                else:
                    mutpb = 0.5
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                # generate new id for mutant
                mutant.id = next(id_gen)
                del mutant.fitness.values
        # combine elite and non elite
        offspring = offspring + list(map(toolbox.clone, elite))
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
        f_bar = sum(fits) / length
        averages.append(f_bar)
        f_global_bar = sum(averages) / len(averages)
        global_max_mean = max(global_max_mean, f_bar)
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - f_bar ** 2) ** 0.5
        min_f = min_ind.fitness.values[0]
        max_f = max_ind.fitness.values[0]
        global_max = max(global_max, max_f)

        history.append(max_f)
        print("  Min %s" % min_f)
        print("  Max %s" % max_f)
        print("  Avg %s" % f_bar)
        print("  Std %s" % std)
        if writer:
            writer.add_scalar('Min Fitness', min_f, g)
            writer.add_scalar('Max Fitness', max_f, g)
            writer.add_scalar('Avg Fitness', f_bar, g)
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
        outcomes["avg"].append(f_bar)
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

            # min max line plot
            plt.plot(range(len(outcomes['avg'])), outcomes['avg'], 'b-', label=ssl_task)
            plt.plot(range(len(outcomes['max'])), outcomes['max'], 'b-')
            plt.plot(range(len(outcomes['max'])), outcomes['min'], 'b-')
            plt.fill_between(range(len(outcomes['avg'])), outcomes['min'], outcomes['max'], color='b',
                             alpha=0.2)
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.xticks = (range(len(outcomes['avg'])))
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(plot_dir, f"min_max.png"))
            plt.clf()

            # scattered boxplot
            data = pd.DataFrame(outcomes["pop_vals"], columns=["Generation", "Fitness"])
            sns.boxplot(data=data, x="Generation", y="Fitness", color='white')
            for i, row in data.iterrows():
                plt.scatter(np.random.normal(row["Generation"], 0.04), row["Fitness"], alpha=0.7, color='skyblue')
            plt.savefig(os.path.join(plot_dir, f"scatter_boxplot.png"))
            plt.clf()

    with open(os.path.join(exp_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes, f)

def GA_mo(pop_size, num_generations,
                             cxpb1 =  0.2,
                             mutpb1 = 0.5,
                             cxpb2 =  0.2,
                             mutpb2 = 0.5,
                             crossover = "PMX",
                             selection = "SUS",
                             dataset="Cifar10",
                             backbone="tinyCNN_backbone",
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
                             adaptive_pb1=None,
                             adaptive_pb2=None,
                             patience = -1,
                             discrete_intensity=False,
                             eval_method="final test",
                             ssl_tasks="v1"
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
    toolbox.register("gen_aug", chromosome_generator_mo(length=chromosome_length,
                                                     discrete=discrete_intensity,
                                                     ssl_tasks=ssl_tasks
                                                     ))
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.gen_aug)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    eval = fitness.fitness_function_mo(dataset=dataset,
                                     backbone=backbone,
                                     ssl_epochs=ssl_epochs,
                                     ssl_batch_size=ssl_batch_size,
                                     evaluate_downstream_method=evaluate_downstream_method,
                                     evaluate_downstream_kwargs=evaluate_downstream_kwargs,
                                     device=device,
                                     seed=seed,
                                       eval_method=eval_method)
    toolbox.register("evaluate", eval)
    if crossover == "PMX":
        toolbox.register("mate", PMX_mo)
    # elif crossover == "twopoint":
    #     toolbox.register("mate", tools.cxTwoPoint)
    # elif crossover == "onepoint":
    #     toolbox.register("mate", tools.cxOnePoint)
    elif crossover == 'onepoint_feas':
        toolbox.register("mate", onepoint_feas_mo)
    else:
        raise ValueError(f"invalid crossover ({crossover})")
    toolbox.register("mutate", mutate.mutGaussianChoice_mo)
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

    f_bar = sum([f[0] for f in fitnesses]) / len(fitnesses)
    global_max_mean = f_bar
    max_f = max_ind.fitness.values[0]
    global_max = max_f
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
        if adaptive_pb1:
            if adaptive_pb1 == "halving":
                # drop_rate = 0.5, gen_drop = 3
                if not g % 3 and g != 0:
                    mutpb1/=2
            elif adaptive_pb1 == "generational":
                cxpb1 = 1 - ((g + 1) / num_generations)
                mutpb1 = ((g + 1) / num_generations)
            elif adaptive_pb1 in ["AGA", "GAGA", "GAGA_V2"]:
                pass
            else:
                raise ValueError(f"invalid adaptive_pb1 value: {adaptive_pb1}")

        if adaptive_pb2:
            if adaptive_pb2 == "halving":
                # drop_rate = 0.5, gen_drop = 3
                if not g % 3 and g != 0:
                    mutpb2/=2
            elif adaptive_pb2 == "generational":
                cxpb2 = 1 - ((g + 1) / num_generations)
                mutpb2 = ((g + 1) / num_generations)
            elif adaptive_pb2 in ["AGA", "GAGA", "GAGA_V2"]:
                pass
            else:
                raise ValueError(f"invalid adaptive_pb2 value: {adaptive_pb2}")
        # Apply crossover and mutation on the offspring
        # split list in two
        children = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if adaptive_pb1 == "AGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb1 = (max_f - f_p) / (max_f - f_bar)
                else:
                    cxpb1 = 1
            elif adaptive_pb1 == "GAGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= global_max_mean:
                    cxpb1 = (global_max - f_p) / (global_max - global_max_mean)
                else:
                    cxpb1 = 1

            elif adaptive_pb1 == "GAGA_V2":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb1 = (global_max - f_p) / (global_max - f_global_bar)
                else:
                    cxpb1 = 1

            if adaptive_pb2 == "AGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb2 = (max_f - f_p) / (max_f - f_bar)
                else:
                    cxpb2 = 1
            elif adaptive_pb2 == "GAGA":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= global_max_mean:
                    cxpb2 = (global_max - f_p) / (global_max - global_max_mean)
                else:
                    cxpb2 = 1

            elif adaptive_pb2 == "GAGA_V2":
                f_p = max([child1.fitness.values[0], child2.fitness.values[0]])
                if f_p >= f_bar:
                    cxpb2 = (global_max - f_p) / (global_max - f_global_bar)
                else:
                    cxpb2 = 1

            if random.random() < cxpb1:

                toolbox.mate(child1, child2)
                # generate new ids for children
                child1.id = next(id_gen)
                child2.id = next(id_gen)
                children.append(child1.id)
                children.append(child2.id)
                del child1.fitness.values
                del child2.fitness.values
            # D4: cx of ssl gene
            if random.random() < cxpb2:
                # swap ssl tasks
                c2_task = child2[0]
                child2[0] = child1[0]
                child1[0] = c2_task

        for mutant in offspring:
            if adaptive_pb1 == "AGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= f_bar:
                    mutpb1 = (0.5 * (max_f - mutant.fitness.values[0])) / (max_f - f_bar)
                else:
                    mutpb1 = 0.5
            elif adaptive_pb1 == "GAGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= global_max_mean:
                    mutpb1 = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - global_max_mean)
                else:
                    mutpb1 = 0.5
            elif adaptive_pb1 == "GAGA_V2":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= f_bar:
                    mutpb1 = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - f_global_bar)
                else:
                    mutpb1 = 0.5

            if adaptive_pb2 == "AGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= f_bar:
                    mutpb2 = (0.5 * (max_f - mutant.fitness.values[0])) / (max_f - f_bar)
                else:
                    mutpb1 = 0.5
            elif adaptive_pb2 == "GAGA":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= global_max_mean:
                    mutpb2 = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - global_max_mean)
                else:
                    mutpb2 = 0.5
            elif adaptive_pb2 == "GAGA_V2":
                # if child was just created this round, mutate
                if mutant.id in children:
                    continue
                # modify mutpb
                if mutant.fitness.values[0] >= f_bar:
                    mutpb2 = (0.5 * (global_max - mutant.fitness.values[0])) / (global_max - f_global_bar)
                else:
                    mutpb2 = 0.5

            if random.random() < mutpb1:
                toolbox.mutate(mutant)
                # generate new id for mutant
                mutant.id = next(id_gen)
                del mutant.fitness.values
            # D5: mutation of ssl gene
            # randomly switch SSL task
            if random.random() < mutpb2:
                mutant.ssl_task = random.choice(SSL_TASKS[ssl_tasks])


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
        f_bar = sum(fits) / length
        global_max_mean = max(global_max_mean, f_bar)
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - f_bar ** 2) ** 0.5
        min_f = min_ind.fitness.values[0]
        max_f = max_ind.fitness.values[0]
        global_max = max(global_max, max_f)
        history.append(max_f)
        print("  Min %s" % min_f)
        print("  Max %s" % max_f)
        print("  Avg %s" % f_bar)
        print("  Std %s" % std)
        if writer:
            writer.add_scalar('Min Fitness', min_f, g)
            writer.add_scalar('Max Fitness', max_f, g)
            writer.add_scalar('Avg Fitness', f_bar, g)
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
        outcomes["avg"].append(f_bar)
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

            # min max line plot
            plt.plot(range(len(outcomes['avg'])), outcomes['avg'], 'b-')
            plt.plot(range(len(outcomes['max'])), outcomes['max'], 'b-')
            plt.plot(range(len(outcomes['max'])), outcomes['min'], 'b-')
            plt.fill_between(range(len(outcomes['avg'])), outcomes['min'], outcomes['max'], color='b',
                             alpha=0.2)
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.xticks = (range(len(outcomes['avg'])))
            plt.savefig(os.path.join(plot_dir, f"min_max.png"))
            plt.clf()

            # scattered boxplot
            data = pd.DataFrame(outcomes["pop_vals"], columns=["Generation", "Fitness"])
            sns.boxplot(data=data, x="Generation", y="Fitness", color='white')
            for i, row in data.iterrows():
                plt.scatter(np.random.normal(row["Generation"], 0.04), row["Fitness"], alpha=0.7, color='skyblue')
            plt.savefig(os.path.join(plot_dir, f"scatter_boxplot.png"))
            plt.clf()
    with open(os.path.join(exp_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes, f)