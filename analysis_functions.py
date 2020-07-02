import copy

import pandas as pd
import numpy as np
import re
import pickle
import math
import random
from tabulate import tabulate
from scipy.stats import kendalltau
from analysis_utils import get_pickle

DEBUG = False


def print_score_ranges(sd):
    metrics_list = get_metrics_list(sd)
    print_list = []
    headers = ["min", "25-perc", "median", "75-perc", "max", "mean"]
    for m in metrics_list:
        scores = [s['scores'][m] for d in sd.values() for s in d['system_summaries'].values()]
        print_list.append([m,
                           np.min(scores),
                           np.percentile(scores, 25),
                           np.median(scores),
                           np.percentile(scores, 75),
                           np.max(scores),
                           np.mean(scores)])
    print(tabulate(print_list, headers=headers, floatfmt=".6f", tablefmt="simple"))


def get_metrics_list(sd):
    """
    Does each system summary dict have same all_metrics?
    :param sd: scores dict
    :return: list of all_metrics in the scores dict
    """
    metrics_tuple_set = set(
        [tuple(sorted(list(x['scores'].keys())))
         for d in sd.values() for x in d['system_summaries'].values()])
    assert len(metrics_tuple_set) == 1, (metrics_tuple_set, "all system summary score dicts should have the same set of all_metrics")
    metrics_list = list(list(metrics_tuple_set)[0])
    return metrics_list
