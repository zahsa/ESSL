import click
import sys
import os
import datetime
from essl.GA import GA, GA_mo

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
@click.option("--eval_method", default="final test", type=str, help="[final test, best val test, best val]")
def GA_cli(pop_size, num_generations,
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
                             eval_method
                             ):
    # save args
    with open(os.path.join(exp_dir, "params.txt"), "w") as f:
         f.write(f"date: {datetime.datetime.now()}\n")
         for a1, a2 in zip(sys.argv[1::2], sys.argv[2::2]):
             f.write(a1 + " " + a2+"\n")

    # save environment
    os.system(f"pip freeze > {os.path.join(exp_dir, 'env.txt')}")
    GA(pop_size=pop_size,
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
         eval_method=eval_method
         )


@click.command()
@click.option("--pop_size", type=int, help="size of population")
@click.option("--num_generations",type=int, help="number of generations")
@click.option("--cxpb1", default=0.2,type=float, help="probability of crossover for aug")
@click.option("--mutpb1", default=0.5,type=float, help="probability of mutation for aug")
@click.option("--cxpb2", default=0.2,type=float, help="probability of crossover for ssl task")
@click.option("--mutpb2", default=0.5,type=float, help="probability of mutation for ssl task")
@click.option("--crossover", default="PMX",type=str, help="type of crossover (PMX, twopoint, onepoint)")
@click.option("--selection", default="SUS",type=str, help="type of selection (SUS, tournament)")
@click.option("--dataset", default="Cifar10",type=str, help="data set to use (Cifar10, )")
@click.option("--backbone", default="ResNet18_backbone",type=str, help="backbone to use (ResNet18_backbone, tinyCNN_backbone, largerCNN_backbone)")
@click.option("--ssl_epochs", default=10, type=int, help="number of epochs for ssl task")
@click.option("--ssl_batch_size", default=256, type=int, help="batch size for ssl task")
@click.option("--evaluate_downstream_method", default="finetune", type=str, help="method of evaluation of ssl representation (finetune)")
@click.option("--device", default="cuda", type=str, help="device for torch (cuda, cpu)")
@click.option("--exp_dir", default="./", type=str, help="path to save experiment results")
@click.option("--use_tensorboard", default=True, type=bool, help="whether to use tensorboard or not")
@click.option("--save_plots", default=True, type=bool, help="whether to save plots or not")
@click.option("--chromosome_length", default=5, type=int, help="number of genes in chromosome")
@click.option("--num_elite", default=0, type=int, help="number of elite chromosomes")
@click.option("--adaptive_pb1", default=None, type=str, help="halving, generational, AGA")
@click.option("--adaptive_pb2", default=None, type=str, help="halving, generational, AGA")
@click.option("--patience", default=-1, type=int, help="number of non-improving generations before early stopping")
@click.option("--discrete_intensity", default=False, type=bool, help="whether or not to use discrete intensity vals")
@click.option("--eval_method", default="final test", type=str, help="[final test, best val test, best val]")
@click.option("--ssl_tasks", default="v1", type=str, help="[v1, v2]")
def GA_mo_cli(pop_size, num_generations,
                             cxpb1,
                             mutpb1,
                             cxpb2,
                             mutpb2,
                             crossover,
                             selection,
                             dataset,
                             backbone,
                             ssl_epochs,
                             ssl_batch_size,
                             evaluate_downstream_method,
                             device,
                             exp_dir,
                             use_tensorboard,
                             save_plots,
                             chromosome_length,
                             num_elite,
                             adaptive_pb1,
                             adaptive_pb2,
                             patience,
                             discrete_intensity,
                             eval_method,
                             ssl_tasks
                             ):
     # save args
     with open(os.path.join(exp_dir, "params.txt"), "w") as f:
          f.write(f"date: {datetime.datetime.now()}\n")
          for a1, a2 in zip(sys.argv[1::2], sys.argv[2::2]):
               f.write(a1 + " " + a2 + "\n")

     # save environment
     os.system(f"pip freeze > {os.path.join(exp_dir, 'env.txt')}")
     GA_mo(pop_size=pop_size,
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
         num_elite=num_elite,
         adaptive_pb1=adaptive_pb1,
         adaptive_pb2=adaptive_pb2,
         patience=patience,
         discrete_intensity=discrete_intensity,
         eval_method=eval_method,
         ssl_tasks=ssl_tasks
         )

if __name__ == "__main__":
    import time
    t1 = time.time()
    GA(pop_size=3,
         ssl_epochs=1,
         num_generations=2,
         ssl_task = "SimSiam",
         backbone="largerCNN_backbone",
         exp_dir=r"/home/noah/ESSL/exps/testing/elite",
         use_tensorboard=False,
         evaluate_downstream_kwargs={"num_epochs":1},
         crossover="PMX",
         adaptive_pb="GAGA",
         eval_method="final test",
         device="cuda",
         num_elite=1
         )
    print(f"GA TOOK {time.time()-t1} to run")
    # t1 = time.time()
    # GA_mo(pop_size=2,
    #    ssl_epochs=1,
    #    num_generations=3,
    #    backbone="largerCNN_backbone",
    #    exp_dir=r"/home/noah/ESSL/exps/testing/merge_essl",
    #    use_tensorboard=False,
    #    evaluate_downstream_kwargs={ "num_epochs": 1 },
    #    crossover="PMX",
    #    adaptive_pb="GAGA",
    #    use_test_acc=False,
    #    device="cuda"
    #    )
    # print(f"GA_mo TOOK {time.time() - t1} to run")

