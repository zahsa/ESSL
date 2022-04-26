import ops
import random

class chromosome_generator:
    def __init__(self, augmentations=ops.DEFAULT_OPS, length=5):
        """
        :param augmentations: dict containing operation, magnitude pairs-
        """
        self.length = length
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
            # gen a float or int based on range types
            (k, random.uniform(*self.augmentations[k]) if isinstance(self.augmentations[k][0], float) else random.randint(*self.augmentations[k]))
            for k in random.sample(list(self.augmentations.keys()), self.length)
        ]

if __name__ == "__main__":
    c = chromosome_generator()
    print(c())
    import pdb;pdb.set_trace()

