from essl.GA import GA, GA_mo

if __name__ == "__main__":
    GA(
        pop_size = 15,
        num_generations = 10,
        cxpb = 0.8,
        mutpb = 0.8,
        dataset = 'Cifar10',
        backbone = 'largerCNN_backbone',
        ssl_task = 'SwaV',
        ssl_epochs = 10,
        ssl_batch_size = 256,
        evaluate_downstream_method = 'finetune',
        device = 'cuda',
        exp_dir = './',
        use_tensorboard = True,
        save_plots = True,
        crossover = 'PMX',
        chromosome_length = 3,
        selection = 'roulette',
        adaptive_pb = 'AGA',
        eval_method = 'best val test',
        num_elite = 2,
        
       )