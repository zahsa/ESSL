from essl.GA import GA, GA_mo

if __name__ == "__main__":
    GA(
        pop_size=15,
        num_generations=10,
        cxpb=0.8,
        mutpb=0.8,
        crossover="PMX",
        selection="SUS",
        dataset="Cifar10",
        backbone="ResNet18_backbone",
        ssl_task="BYOL",
        ssl_epochs=10,
        ssl_batch_size=256,
        evaluate_downstream_method="finetune",
        evaluate_downstream_kwargs={ },
        device="cuda",
        exp_dir="./",
        use_tensorboard=True,
        save_plots=True,
        chromosome_length=3,
        seed=10,
        num_elite=2,
        adaptive_pb="AGA",
        patience=-1,
        discrete_intensity=False,
        eval_method="best val test"
       )