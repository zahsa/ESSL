from essl.utils import train_model_longer_fr_scratch
import os
import glob
import pandas as pd


def run_exp(chromosome, algo, seed, save_dir, train_kwargs, ssl_epochs= [50, 100, 1000], downstream_epochs= [50]):
    
    for se in ssl_epochs:
        for de in downstream_epochs:
            exp_dir = os.path.join(save_dir, f"{se}_{de}")
            if not os.path.isdir(exp_dir):
                os.mkdir(exp_dir)
            else:
                continue

            print(f"algorithm: {algo}")
            print(f"dataset: {train_kwargs['dataset']}")
            print(f"training ssl e: {se}")
            print(f"training downstream e: {de}")
            train_model_longer_fr_scratch(chromosome,
                                            seed,
                                            backbone="largerCNN_backbone",
                                            ssl_task=algo,
                                            ssl_epochs=se,
                                            downstream_epochs=de,
                                            save_dir=exp_dir,
                                            **train_kwargs)

if __name__ == "__main__":
    best_chromos = pd.read_csv("/home/noah/ESSL/exps/Analysis/Final Results/best_chromos/best_chromos.csv")
    # CIFAR10
    # for _, bc in best_chromos[best_chromos['exp'].apply(lambda x: 'exp6' in x)].iterrows():
    #    chromo = list(bc[['aug1', 'op1', 'aug2', 'op2', 'aug3', 'op3']])
    #    chromo = [chromo[0:2], chromo[2:4], chromo[4:]]
    #    algo = bc['algo']
    #    seed = bc['seed']
    #    train_kwargs = {"dataset":"Cifar10", "ssl_batch_size":32}
    #    save_dir = os.path.join("/home/noah/ESSL/exps/Analysis/train_models_longer/best_overall_acc",
    #                 f"{train_kwargs['dataset']}_{train_kwargs['ssl_batch_size']}_{algo}")
    #    if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #    run_exp(chromo, algo, seed, save_dir, train_kwargs)

    # SVHN
    # for _, bc in best_chromos[best_chromos['exp'].apply(lambda x: 'exp11' in x)].iterrows():
    #    chromo = list(bc[['aug1', 'op1', 'aug2', 'op2', 'aug3', 'op3']])
    #    chromo = [chromo[0:2], chromo[2:4], chromo[4:]]
    #    algo = bc['algo']
    #    seed = bc['seed']
    #    train_kwargs = {"dataset":"SVHN", "ssl_batch_size":32}
    #    save_dir = os.path.join("/home/noah/ESSL/exps/Analysis/train_models_longer/best_overall_acc",
    #                 f"{train_kwargs['dataset']}_{train_kwargs['ssl_batch_size']}_{algo}")
    #    if not os.path.isdir(save_dir):
    #         os.mkdir(save_dir)
    #    run_exp(chromo, algo, seed, save_dir, train_kwargs)
    # BYOL CIFAR10, BS=256
    bc = best_chromos[best_chromos['exp'] == 'exp10_1'].iloc[0]
    chromo = list(bc[['aug1', 'op1', 'aug2', 'op2', 'aug3', 'op3']])
    chromo = [chromo[0:2], chromo[2:4], chromo[4:]]
    algo = bc['algo']
    seed = bc['seed']
    train_kwargs = {"dataset":"SVHN", "ssl_batch_size":256}
    save_dir = os.path.join("/home/noah/ESSL/exps/Analysis/train_models_longer/best_overall_acc",
                f"{train_kwargs['dataset']}_{train_kwargs['ssl_batch_size']}_{algo}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    run_exp(chromo, algo, int(seed), save_dir, train_kwargs)