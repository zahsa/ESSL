"""
perturbation of intensities
    - must remain within the feasible space
    - pre or post
"""

class MutateOperators:

    def __init__(self,chromosome,fitness,avg_fitness,mute_rate,crossover_rate):
        """
        """
        self.chromosome = chromosome
        self.fitness = fitness
        self.avg_fitness = avg_fitness
        self.mute_rate = mute_rate
        self.crossover_rate = crossover_rate
        self.parents = parents

    def mutateOp1(self):
        if self.chrom_fitness < self.avg_fitness:
            mute_rate *= 2
        else:
            mute_rate /= 2

        if np.random.rand() <= mute_rate:
            noise = np.random.uniform(0, 1, 1)
            self.chromosome['intensity'] += noise











