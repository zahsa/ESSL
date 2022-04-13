import ops
import random

class chromosome_generator:
    def __init__(self, augmentations=ops.DEFAULT_OPS):
        """
        :param augmentations: dict containing operation, magnitude pairs-
        """
        self.augmentations = augmentations
        # encode augmentations as integer
        self.pheno2geno = {
            a:i for i, a in enumerate(self.augmentations)
        }
        # get augmentations back to integer
        self.geno2pheno = {
            i: a for i, a in enumerate(self.augmentations)
        }
        
    def __call__(self):
        # representation = random permutation and random intensity
        return [
            (self.pheno2geno[k], random.uniform(*self.augmentations[k]))
            for k in random.sample(list(self.augmentations.keys()), len(self.augmentations))
        ]

if __name__ == "__main__":
    c = chromosome_generator()
    import pdb;pdb.set_trace()

