from essl import ops
import random
from itertools import permutations
"""
New chromosome representation:

contains both intensities and the ssl task


"""

SSL_TASKS = [
                "NNCLR",
                "SimCLR",
                "SwaV",
                "BYOL"
]

class chromo(list):
    def __init__(self, ssl_task
                 ):
        self.ssl_task = ssl_task
        #self.augmentation = augmentation

class chromosome_generator:
    def __init__(self, augmentations=ops.DEFAULT_OPS,
                 length=5, seed=10, discrete=False, intensity_increments=10):
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
    # D1: add ssl task as part of chromosome
    def __call__(self):
        chromosome = chromo(ssl_task=random.choice(SSL_TASKS))
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
            # augmentation = [
            # # gen a float or int based on range types
            # [k, random.uniform(*self.augmentations[k])
            #             if isinstance(self.augmentations[k][0], float)
            #             else random.randint(*self.augmentations[k])]
            # for k in random.sample(list(self.augmentations), self.length)
            # ]
        return chromosome


if __name__ == "__main__":
    c = chromosome_generator()
    print(c())
    import pdb;
    pdb.set_trace()
