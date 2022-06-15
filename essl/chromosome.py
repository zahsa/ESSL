from essl import ops
import random
from itertools import permutations


class chromosome_generator:
    def __init__(self, augmentations=ops.DEFAULT_OPS, length=5, seed=10):
        """
        :param augmentations: dict containing operation, magnitude pairs-
        """
        self.length = length
        self.augmentations = augmentations
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
        return [
            # gen a float or int based on range types
            [k, random.uniform(*self.augmentations[k])
                        if isinstance(self.augmentations[k][0], float)
                        else random.randint(*self.augmentations[k])]
            for k in random.sample(list(self.augmentations), self.length)
        ]


if __name__ == "__main__":
    c = chromosome_generator()
    print(c())
    import pdb;
    pdb.set_trace()
