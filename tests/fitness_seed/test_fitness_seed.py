import unittest
import torch
from essl.fitness import fitness_function
from essl.chromosome import chromosome_generator
class TestSSLFixedTrainingSeed(unittest.TestCase):
    def test_ssl(self):
        chromo = chromosome_generator(seed=10)()
        fitness = fitness_function(dataset="Cifar10",
                                 exp_dir="./",
                                 backbone="largerCNN_backbone",
                                 ssl_task="NNCLR",
                                 ssl_epochs=1,
                                 ssl_batch_size=32,
                                 evaluate_downstream_method="finetune",
                                 evaluate_downstream_kwargs={ "num_epochs": 1},
                                 device= "cuda",
                                 seed=10,
                                 eval_method = "best val test")
        outcomes1 = fitness(chromo, return_losses=True)
        outcomes2 = fitness(chromo, return_losses=True)
        for metric1, metric2 in zip(outcomes1, outcomes2):
            import pdb;pdb.set_trace()
            print("*************************************************")
            print(metric1)
            print("*************************************************")
            print(metric2)
            # truth = torch.eq(metric1, metric2)
            # print(truth)
            # print(sum(truth))
            # assert sum(truth) == 0





if __name__ == '__main__':
    unittest.main()