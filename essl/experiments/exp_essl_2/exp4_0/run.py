import random
import os
from essl.GA import GA, GA_mo
import datetime

num_seeds = 3
pop_size = 15
num_generations = 10
cxpb1 = 0.8
cxpb2 = 0.5
mutpb1 = 0.8
mutpb2 = 0.2
crossover = "PMX"
selection = "roulette"
dataset = "Cifar10"
backbone = "largerCNN_backbone"
ssl_epochs = 10
ssl_batch_size = 32
evaluate_downstream_method = "finetune"
device = "cuda"
exp_dir = "./"
use_tensorboard = True
save_plots = True
chromosome_length = 3
num_elite = 2
adaptive_pb1 = "AGA"
patience = -1
discrete_intensity = False
eval_method = "best val test"
ssl_tasks = "v6"

args = {'num_seeds':num_seeds,
    'pop_size':pop_size,
    'num_generations':num_generations,
    'cxpb1':cxpb1,
    'mutpb1':mutpb1,
    'cxpb2':cxpb2,
    'mutpb2':mutpb2,
    'crossover':crossover,
    'selection':selection,
    'dataset':dataset,
    'backbone':backbone,
    'ssl_epochs':ssl_epochs,
    'ssl_batch_size':ssl_batch_size,
    'evaluate_downstream_method':evaluate_downstream_method,
    'device':device,
    'exp_dir':exp_dir,
    'use_tensorboard':use_tensorboard,
    'save_plots':save_plots,
    'chromosome_length':chromosome_length,
    'num_elite':num_elite,
    'adaptive_pb1':adaptive_pb1,
    'patience':patience,
    'discrete_intensity':discrete_intensity,
    'eval_method':eval_method,
    'ssl_tasks':ssl_tasks}

if __name__ == "__main__":
    for seed in random.sample(range(10), num_seeds):
        exp_seed_dir = os.path.join(exp_dir, str(seed))
        if not os.path.isdir(exp_seed_dir):
            os.mkdir(exp_seed_dir)

        # save environment
        os.system(f"pip freeze > {os.path.join(exp_seed_dir, 'env.txt')}")
        with open(os.path.join(exp_seed_dir, "params.txt"), "w") as f:
            f.write(f"date: {datetime.datetime.now()}\n")
            for a1, a2 in args.items():
                f.write("--"+a1 + " " + str(a2) + "\n")

        GA_mo(
            pop_size=pop_size,
            num_generations=num_generations,
            cxpb1=cxpb1,
            mutpb1=mutpb1,
            cxpb2=cxpb2,
            mutpb2=mutpb2,
            crossover=crossover,
            selection=selection,
            dataset=dataset,
            backbone=backbone,
            ssl_epochs=ssl_epochs,
            ssl_batch_size=ssl_batch_size,
            evaluate_downstream_method=evaluate_downstream_method,
            device=device,
            exp_dir=exp_dir,
            use_tensorboard=use_tensorboard,
            save_plots=save_plots,
            chromosome_length=chromosome_length,
            seed=seed,
            num_elite=num_elite,
            adaptive_pb1=adaptive_pb1,
            patience=patience,
            discrete_intensity=discrete_intensity,
            eval_method=eval_method,
            ssl_tasks=ssl_tasks
           )
