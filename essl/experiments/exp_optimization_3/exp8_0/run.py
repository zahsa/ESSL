import random
import os
import datetime
from essl.GA import GA, GA_mo

num_seeds = 10
pop_size = 15
num_generations = 10
cxpb = 0.8
mutpb = 0.8
crossover = "PMX"
selection = "roulette"
dataset = "Cifar10"
backbone = "largerCNN_backbone"
ssl_task = "SwaV"
ssl_epochs = 10
ssl_batch_size = 256
evaluate_downstream_method = "finetune"
evaluate_downstream_kwargs = { }
device = "cuda"
exp_dir = "./"
use_tensorboard = True
save_plots = True
chromosome_length = 3
num_elite = 2
adaptive_pb = "AGA"
patience = -1
discrete_intensity = False
eval_method = "best val test"

args = {'num_seeds':num_seeds,
    'pop_size':pop_size,
    'num_generations':num_generations,
    'cxpb':cxpb,
    'mutpb':mutpb,
    'crossover':crossover,
    'selection':selection,
    'dataset':dataset,
    'backbone':backbone,
    'ssl_task':ssl_task,
    'ssl_epochs':ssl_epochs,
    'ssl_batch_size':ssl_batch_size,
    'evaluate_downstream_method':evaluate_downstream_method,
    'device':device,
    'exp_dir':exp_dir,
    'use_tensorboard':use_tensorboard,
    'save_plots':save_plots,
    'chromosome_length':chromosome_length,
    'num_elite':num_elite,
    'adaptive_pb':adaptive_pb,
    'patience':patience,
    'discrete_intensity':discrete_intensity,
    'eval_method':eval_method}
if __name__ == "__main__":
    for seed in random.sample(range(10), num_seeds):
        exp_seed_dir = os.path.join(exp_dir, str(seed))
        if not os.path.isdir(exp_seed_dir):
            os.mkdir(exp_seed_dir)

        with open(os.path.join(exp_seed_dir, "params.txt"), "w") as f:
            f.write(f"date: {datetime.datetime.now()}\n")
            for a1, a2 in args.items():
                f.write("--"+a1 + " " + str(a2) + "\n")

        # save environment
        os.system(f"pip freeze > {os.path.join(exp_seed_dir, 'env.txt')}")
        GA(
            pop_size=pop_size,
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
            evaluate_downstream_kwargs=evaluate_downstream_kwargs,
            device=device,
            exp_dir=exp_seed_dir,
            use_tensorboard=use_tensorboard,
            save_plots=save_plots,
            chromosome_length=chromosome_length,
            seed=seed,
            num_elite=num_elite,
            adaptive_pb=adaptive_pb,
            patience=patience,
            discrete_intensity=discrete_intensity,
            eval_method=eval_method
           )