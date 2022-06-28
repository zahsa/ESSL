import multiprocessing
import time

from essl.chromosome import chromosome_generator
from functools import partial

def dist_fit(gpu, sub_pop, eval_func):
    outcomes = list(map(partial(eval_func, device=f"cuda:{gpu}"), sub_pop))
    #outcomes = list(map(eval_func, sub_pop))
    #outcomes = gpu
    return outcomes