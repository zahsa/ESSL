import random


def PMX_inner(s, t, size, values1, values2):
    _map1 = { }
    _map2 = { }
    # choose two crossover points
    x1 = random.randint(0, size-1)
    x2 = random.randint(x1+1, size)
    offspring = [s.copy(), t.copy()]
    for i in range(x1, x2):

        offspring[0][i] = t[i]
        _map1[t[i][0]] = s[i][0]

        offspring[1][i] = s[i]
        _map2[s[i][0]] = t[i][0]

    # check first part of chromosome before the swap
    for i in range(0, x1):
        while offspring[0][i][0] in _map1:
            offspring[0][i] = values1[_map1[offspring[0][i][0]]]

        while offspring[1][i][0]  in _map2:
            offspring[1][i] = values2[_map2[offspring[1][i][0]]]

    # check after the swap
    for i in range(x2, size):
        while offspring[0][i][0] in _map1:
            offspring[0][i] = values1[_map1[offspring[0][i][0]]]
        while offspring[1][i][0] in _map2:
            offspring[1][i] = values2[_map2[offspring[1][i][0]]]

    return offspring
def PMX(s, t):
    """
    create mapping based on augmentation operator itself,
    does not consider intensity
    :param s:
    :param t:
    :return:
    """

    size = min([len(s), len(t)])
    values1 = { s[i][0]: s[i] for i in range(size) }
    values2 = { t[i][0]: t[i] for i in range(size) }
    offspring = PMX_inner(s, t, size, values1, values2)
    # feasibility check
    parents = [s,t]
    while offspring[0] in parents or offspring[1] in parents or offspring[0] == offspring[1]:
        offspring = PMX_inner(s, t, size, values1, values2)

    return offspring



if __name__ == "__main__":
    from essl.chromosome import chromosome_generator
    c = chromosome_generator()
    chromo1 = c()
    chromo2 = c()

    i, k = PMX(chromo1, chromo2)
    print(i)
    print(k)
    import pdb;pdb.set_trace()
