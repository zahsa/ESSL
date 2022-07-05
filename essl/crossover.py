import random


def PMX_inner(s, t, size, values1, values2):
    _map1 = { }
    _map2 = { }
    # choose two crossover points
    x1 = random.randint(0, size-1)
    x2 = random.randint(x1+1, size)
    offspring = [s.copy(), t.copy()]
    for i in range(x1, x2):
        # swap x1 to x2 of s with x1 to x2 of t
        offspring[0][i] = t[i]
        _map1[t[i][0]] = s[i][0]
        # swap x1 to x2 of t with x1 to x2 of s
        offspring[1][i] = s[i]
        _map2[s[i][0]] = t[i][0]

    # check first part of chromosome before the swap
    for i in range(0, x1):
        while offspring[0][i][0] in _map1:
            offspring[0][i] = values1[_map1[offspring[0][i][0]]]
        while offspring[1][i][0] in _map2:
            offspring[1][i] = values2[_map2[offspring[1][i][0]]]

    # check after the swap
    for i in range(x2, size):
        while offspring[0][i][0] in _map1:
            offspring[0][i] = values1[_map1[offspring[0][i][0]]]
        while offspring[1][i][0] in _map2:
            offspring[1][i] = values2[_map2[offspring[1][i][0]]]
    return offspring
# D1: set num attempts to 5
def PMX(s,t, num_attempts=5):
    """
    create mapping based on augmentation operator itself,
    does not consider intensity
    :param s: parent 1
    :param t: parent 2
    :param num_attempts: number of attempts to cx if still duplicate genes just return parents
    :return:
    """

    size = min([len(s), len(t)])
    # mapping for augmentation operators
    values1 = { s[i][0]: s[i] for i in range(size) }
    values2 = { t[i][0]: t[i] for i in range(size) }
    offspring = PMX_inner(s, t, size, values1, values2)
    # feasibility check
    parents = [s,t]
    c = 0
    # D2: fix logic for feasibility check
    while offspring[0] in parents or offspring[1] in parents or offspring[0] == offspring[1]\
            or len(offspring[0]) != len(set([i[0] for i in offspring[0]])) or len(offspring[1]) != len(set([i[0] for i in offspring[1]])):
        offspring = PMX_inner(s, t, size, values1, values2)
        c+=1
        if c == num_attempts: # try 5 times if it doesnt work, then just return parents
            offspring = parents
            break
    s[:], t[:] = offspring
    return offspring

# D1: change name to onepoint_feas
def onepoint_feas(ind1, ind2, n_tries=5):
    size = min(len(ind1), len(ind2))
    child1, child2 = ind1.copy(), ind2.copy()
    cxpoint = random.randint(1, size - 1)
    child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
    total = 0
    # if there are duplicate operators in ind1 or ind 2
    while len(set([g[0] for g in child1])) < len(child1) or \
          len(set([g[0] for g in child2])) < len(child2):
        cxpoint = random.randint(1, size - 1)
        child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
        total+=1
        if total == n_tries:
            child1 = ind1
            child2 = ind2
            break
    ind1[:], ind2[:] = child1[:], child2[:]
    return ind1, ind2

if __name__ == "__main__":
    from essl.chromosome import chromosome_generator
    c = chromosome_generator()
    chromo1 = c()
    chromo2 = c()

    i, k = PMX(chromo1, chromo2)
    print(i)
    print(k)
    import pdb;pdb.set_trace()
