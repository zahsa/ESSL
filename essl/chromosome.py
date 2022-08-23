from essl import ops
import random
from itertools import permutations
import numpy as np

SSL_TASKS = {
    "v1": [
                    "NNCLR",
                    "SimCLR",
                    "SwaV",
                    "BYOL"
    ],
    "v2": [
                    "NNCLR",
                    "SwaV",
                    "BYOL"
    ],
    "v3": [
                    "NNCLR",
                    "SwaV",
                    "BYOL",
                    "SimSiam",
                    "BarlowTwins"
    ],
    "v4": [
                    "NNCLR",
                    "SwaV",
                    "BYOL",
                    "SimSiam",
                    "BarlowTwins",
                    "MoCo"
    ],
    "v5":[
                    "NNCLR",
                    "SwaV",
                    "BYOL",
                    "SimSiam",
                    "MoCo"
    ],
    "v6":[
                    "NNCLR",
                    "SwaV",
                    "BYOL",
                    "SimSiam"
    ]
}

class chromosome_generator:
    def __init__(self, augmentations=ops.DEFAULT_OPS,
                 length=5, discrete=False, seed=10, intensity_increments=10):
        """
        :param augmentations: dict containing operation, magnitude pairs-
        """
        self.length = length
        self.augmentations = augmentations
        self.discrete = discrete
        self.intensity_increments = intensity_increments
        random.seed(seed)
        # encode augmentations as integer
        self.pheno2geno = {
            a: i for i, a in enumerate(self.augmentations)
        }
        # get augmentations back to integer
        self.geno2pheno = {
            i: a for i, a in enumerate(self.augmentations)
        }

    @property
    def search_space(self):
        return permutations(self.augmentations, self.length)

    def gen_search_space(self):
        return [
                    [
                        [k, random.uniform(*self.augmentations[k])
                            if isinstance(self.augmentations[k][0], float)
                            else random.randint(*self.augmentations[k])]
                            for k in chromo
                    ]
                for chromo in self.search_space
        ]
    def __call__(self):
        # representation = random permutation and random intensity
        if self.discrete:
            chromosome = []
            for k in random.sample(list(self.augmentations), self.length):
                increment = abs(self.augmentations[k][1] - self.augmentations[k][0]) / self.intensity_increments
                if isinstance(self.augmentations[k][0], float):
                    intensity = float(round(random.choice(np.arange(*self.augmentations[k], increment, dtype=float)), 2))
                else:
                    intensity = int(random.choice(np.arange(*self.augmentations[k], increment, dtype=int)))
                chromosome.append([k, intensity])
        else:
            chromosome =  [
            # gen a float or int based on range types
            [k, random.uniform(*self.augmentations[k])
                        if isinstance(self.augmentations[k][0], float)
                        else random.randint(*self.augmentations[k])]
            for k in random.sample(list(self.augmentations), self.length)
        ]

        return chromosome

class chromosome_generator_mo:
    def __init__(self, augmentations=ops.DEFAULT_OPS,
                 length=5, seed=10, discrete=False, intensity_increments=10, ssl_tasks = "v1"):
        """
        :param augmentations: dict containing operation, magnitude pairs-
        """
        self.length = length
        self.augmentations = augmentations
        self.discrete = discrete
        self.intensity_increments = intensity_increments
        self.ssl_tasks = ssl_tasks
        random.seed(seed)
        # encode augmentations as integer
        self.pheno2geno = {
            a: i for i, a in enumerate(self.augmentations)
        }
        # get augmentations back to integer
        self.geno2pheno = {
            i: a for i, a in enumerate(self.augmentations)
        }

    @property
    def search_space(self):
        raise NotImplementedError

    def gen_search_space(self):
        raise NotImplementedError

    def __call__(self):
        # chromosome = chromo(ssl_task=random.choice(SSL_TASKS))
        chromosome= [random.choice(SSL_TASKS[self.ssl_tasks])]
        # representation = random permutation and random intensity
        if self.discrete:
            for k in random.sample(list(self.augmentations), self.length):
                increment = abs(self.augmentations[k][1] - self.augmentations[k][0]) / self.intensity_increments
                if isinstance(self.augmentations[k][0], float):
                    intensity = float(round(random.choice(np.arange(*self.augmentations[k], increment, dtype=float)), 2))
                else:
                    intensity = int(random.choice(np.arange(*self.augmentations[k], increment, dtype=int)))
                chromosome.append([k, intensity])
        else:
            for k in random.sample(list(self.augmentations), self.length):
                if isinstance(self.augmentations[k][0], float):
                    chromosome.append([k, random.uniform(*self.augmentations[k])])
                else:
                    chromosome.append([k, random.randint(*self.augmentations[k])])

        return chromosome

if __name__ == "__main__":
    c = chromosome_generator()
    print(c())

