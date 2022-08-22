import random
import os
import datetime
from essl.GA import GA, GA_mo

pop_size = 15
num_generations = 10
cxpb = 0.8
mutpb = 0.8
dataset = 'Cifar10'
backbone = 'largerCNN_backbone'
ssl_task = 'SwaV'
ssl_epochs = 10
ssl_batch_size = 256
evaluate_downstream_method = 'finetune'
device = 'cuda'
exp_dir = './'
use_tensorboard = True
save_plots = True
crossover = 'PMX'
chromosome_length = 3
selection = 'roulette'
adaptive_pb = 'AGA'
num_seeds = 5
eval_method = 'best val test'
num_elite = 2
aug_ops = 'OPS_NO_FLIP'


args = {
    'pop_size':15,
    'num_generations':10,
    'cxpb':0.8,
    'mutpb':0.8,
    'dataset':'Cifar10',
    'backbone':'largerCNN_backbone',
    'ssl_task':'SwaV',
    'ssl_epochs':10,
    'ssl_batch_size':256,
    'evaluate_downstream_method':'finetune',
    'device':'cuda',
    'exp_dir':'./',
    'use_tensorboard':True,
    'save_plots':True,
    'crossover':'PMX',
    'chromosome_length':3,
    'selection':'roulette',
    'adaptive_pb':'AGA',
    'num_seeds':5,
    'eval_method':'best val test',
    'num_elite':2,
    'aug_ops':'OPS_NO_FLIP',
    
    }

if __name__ == "__main__":
    for seed in range(num_seeds):
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
            
           )