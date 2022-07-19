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
    return s, t

def PMX_mo(s,t, num_attempts=5):
    s[1:], t[1:] = PMX(s[1:],t[1:], num_attempts)
    return s, t

def onepoint_feas(s, t, num_attempts=20):
    size = min(len(s), len(t))
    child1, child2 = s.copy(), t.copy()
    cxpoint = random.randint(1, size - 1)
    child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
    total = 0
    # if there are duplicate operators in s or ind 2
    while len(set([g[0] for g in child1])) < len(child1) or \
          len(set([g[0] for g in child2])) < len(child2):
        child1, child2 = s.copy(), t.copy()
        cxpoint = random.randint(1, size - 1)
        child1[cxpoint:], child2[cxpoint:] = child2[cxpoint:], child1[cxpoint:]
        total+=1
        if total == num_attempts:
            child1 = s
            child2 = t
            break
    s[:], t[:] = child1[:], child2[:]
    return s, t

def onepoint_feas_mo(s,t, num_attempts=5):
    s[1:], t[1:] = onepoint_feas(s[1:],t[1:], num_attempts)
    return s, t

if __name__ == "__main__":
    from essl.chromosome import chromosome_generator_mo
    from copy import deepcopy
    c = chromosome_generator_mo(seed=13)
    chromo1 = c()
    chromo2 = c()
    cc1 = deepcopy(chromo1)
    cc2 = deepcopy(chromo2)
    # for _ in range(10):
    print(chromo1)
    c1, c2 = onepoint_feas_mo(chromo1, chromo2)
    print(chromo1)
    # print(c1, c2)