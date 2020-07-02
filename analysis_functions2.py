import itertools

import pandas as pd
import numpy as np
import re
import math
import random
import scipy
import matplotlib.pyplot as plt
import logging
import pdb
from scipy.stats import kendalltau
from analysis_functions import get_metrics_list
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
from joblib import parallel_backend

DEBUG = False

ref_summ_x_types = ['vocab_by_nwords',
                    'n_words',
                    'n_sents',
                    'n_words_per_sent',
                    'summ_abstractiveness_wrt_ref',
                    'ref_abstractiveness_wrt_doc',
                    ]
ref_summ_x_modifiers = ['doc', 'ref', 'summ', 'ref_summ_diff', 'ref_summ_abs_diff']

summpair_y_types = ['normed_normed_scores_abs_diff', 'normed_normed_scores_diff',
                    'normed_m1_diff_by_m2_diff',
                    'normed_scores_abs_diff', 'normed_scores_diff',
                    'scores_abs_diff', 'scores_diff']

summpair_x_types = ['m_avg_norm', 'm_avg', 'm1', 'm2', 'm1_norm', 'm2_norm']

summ_x_types = ['m_avg_norm', 'm_avg', 'm1', 'm2', 'm1_norm', 'm2_norm']

summ_y_types = ['normed_scores_abs_diff', 'normed_scores_diff',
                'scores_abs_diff', 'scores_diff']

doc_x_types = ['avg__avg_m1_avg_m2', 'avg_m1', 'avg_m2', 'avg__max_m1_max_m2',
               'max_m1', 'max_m2', 'compression', 'coverage', 'density']

more_doc_x_types = ['avg__all_max_m', 'avg__all_avg_m']

doc_y_types = ['ktau', 'pearson', 'spearman', 'm']

metric_remap = {'bert_recall_score': 'BScore',
                'mover_score': 'MS',
                'rouge_1_recall': 'R1',
                'rouge_2_recall': 'R2',
                'rouge_l_recall': 'RL',
                'js-2': 'JS2'}


# plot any Y vs X
def plot_X_Y(X, Y, x_type, y_type, m1, m2, abs_pr_plot_cutoff, fit_line_window, 
             show_y_label=True, show_title=True, show_yticklabels=True, save_fig=False):
    try:
        pearsons_r, p_val = scipy.stats.pearsonr(X, Y)
    except Exception:
        pearsons_r, p_val = -42, -42
    if abs_pr_plot_cutoff:
        if abs(pearsons_r) < abs_pr_plot_cutoff:
            return
    print(f"x:{x_type}\ty:{y_type}\tm1:{m1}\tm2:{m2}\tpr:{pearsons_r:.3f} {p_val:E}\n"
          f"mean x: {np.mean(X)}\tmean y:{np.mean(Y)}")
    fig, ax = plt.subplots()
    sc = plt.scatter(x=X, y=Y, s=1)  # , figsize=(10, 7), linewidth=3)
    data = list(zip(X, Y))
    data.sort(key=lambda t: t[0])
    X, Y = zip(*data)
    x_fit = pd.Series(X).rolling(window=fit_line_window).mean().iloc[fit_line_window - 1:].values
    y_fit = pd.Series(Y).rolling(window=fit_line_window).mean().iloc[fit_line_window - 1:].values
    plt.plot(x_fit, y_fit, color='red')
    # ax.set_xlabel(x_type, fontsize=25)
    y_name = y_type
    if y_type == 'ktau':
        y_name = "Kendall's tau"
    if show_y_label:
        ax.set_ylabel(y_name, fontsize=35)
    mpair_name = '_'.join([metric_remap[m1], metric_remap[m2]])
    if show_title:
        ax.set_title(mpair_name.replace('_', ', '), fontsize=25)
    ax.tick_params(axis='both', labelsize=25)
    if not show_yticklabels:
        ax.set_yticklabels([])

    if save_fig:
        pass
    plt.show()


# plotting graphs where points are summary PAIRS
def get_summpair_each_doc(n, sd):
    """
    n is the number of pairs sampled from EACH DOC
    generate a list of summary-pairs-tuples of the form (doc_id, sysname_1, sysname2) from scores_dict.
    """
    pairs = []
    # n_per_doc = n//len(sd)
    for doc_id, isd in sd.items():
        for _ in range(n):
            pair = random.sample(list(isd['system_summaries'].keys()), 2)
            pairs.append((doc_id, *pair))
    return pairs


def get_summpair_x_y_series(m1, m2, sd, summpairs, x_type, y_type):
    # data = []
    X = []
    Y = []
    for doc_id, sys1, sys2 in summpairs:
        isd = sd[doc_id]
        sumdict1 = sd[doc_id]['system_summaries'][sys1]
        sumdict2 = sd[doc_id]['system_summaries'][sys2]
        ref_summ = isd['ref_summ']
        x_val = get_summpair_x_val(ref_summ, m1, m2, sumdict1, sumdict2, x_type)
        y_val = get_summpair_y_val(ref_summ, m1, m2, sumdict1, sumdict2, y_type)
        X.append(x_val)
        Y.append(y_val)
        print(f"get_summpairs_x_y_series: done {doc_id}/{len(sd)} pairs", end="\r")
    return X, Y


def get_summpair_x_val(ref, m1, m2, sumdict1, sumdict2, x_type):
    assert x_type in summpair_x_types
    if x_type == 'm_avg_norm':
        return np.mean([sumdict1['normed_scores'][m1],
                        sumdict1['normed_scores'][m2],
                        sumdict2['normed_scores'][m1],
                        sumdict2['normed_scores'][m2],
                        ])
    elif x_type == 'm_avg':
        return np.mean([sumdict1['scores'][m1],
                        sumdict1['scores'][m2],
                        sumdict2['scores'][m1],
                        sumdict2['scores'][m2],
                        ])
    elif x_type == 'm1':
        return np.mean([sumdict1['scores'][m1],
                        sumdict2['scores'][m1],
                        ])
    elif x_type == 'm2':
        return np.mean([sumdict1['scores'][m2],
                        sumdict2['scores'][m2],
                        ])
    elif x_type == 'm1_norm':
        return np.mean([sumdict1['normed_scores'][m1],
                        sumdict2['normed_scores'][m1],
                        ])
    elif x_type == 'm2_norm':
        return np.mean([sumdict1['normed_scores'][m2],
                        sumdict2['normed_scores'][m2],
                        ])


def get_summpair_y_val(ref, m1, m2, sumdict1, sumdict2, y_type):
    assert y_type in summpair_y_types
    if y_type == 'normed_normed_scores_abs_diff':
        a = (sumdict1['normed_scores'][m1] - sumdict2['normed_scores'][m1])
        b = (sumdict1['normed_scores'][m2] - sumdict2['normed_scores'][m2])
        return abs((a - b) / (a + b))
    elif y_type == 'normed_normed_scores_diff':
        a = (sumdict1['normed_scores'][m1] - sumdict2['normed_scores'][m1])
        b = (sumdict1['normed_scores'][m2] - sumdict2['normed_scores'][m2])
        return (a - b) / (a + b)
    elif y_type == 'normed_m1_diff_by_m2_diff':
        a = (sumdict1['normed_scores'][m1] - sumdict2['normed_scores'][m1])
        b = (sumdict1['normed_scores'][m2] - sumdict2['normed_scores'][m2])
        return np.log((a / (b + 1e-4)) + 1e-4)
    elif y_type == 'normed_scores_abs_diff':
        return abs(
            (sumdict1['normed_scores'][m1] - sumdict2['normed_scores'][m1]) -
            (sumdict1['normed_scores'][m2] - sumdict2['normed_scores'][m2])
        )
    elif y_type == 'normed_scores_diff':
        return (
                (sumdict1['normed_scores'][m1] - sumdict2['normed_scores'][m1]) -
                (sumdict1['normed_scores'][m2] - sumdict2['normed_scores'][m2])
        )
    elif y_type == 'scores_abs_diff':
        return abs(
            (sumdict1['scores'][m1] - sumdict2['scores'][m1]) -
            (sumdict1['scores'][m2] - sumdict2['scores'][m2])
        )
    elif y_type == 'scores_diff':
        return (
                (sumdict1['scores'][m1] - sumdict2['scores'][m1]) -
                (sumdict1['scores'][m2] - sumdict2['scores'][m2])
        )


def plot_summpair(sd, n_pairs, x_types, y_types, metrics_list=None, abs_pr_plot_cutoff=None, fit_line_window=100):
    if not metrics_list:
        metrics_list = get_metrics_list(sd)

    summpairs = get_summpair_each_doc(n_pairs, sd)

    for x_type in x_types:
        for y_type in y_types:
            plot_summpair_inner(sd=sd, x_type=x_type, y_type=y_type, summpairs=summpairs,
                                metrics_list=metrics_list, abs_pr_plot_cutoff=abs_pr_plot_cutoff,
                                fit_line_window=fit_line_window)


def plot_summpair_inner(sd, x_type, y_type, summpairs,
                        metrics_list=None, abs_pr_plot_cutoff=None,
                        fit_line_window=100):
    for m1_idx, m1 in enumerate(metrics_list):
        for m2 in metrics_list[m1_idx + 1:]:
            X, Y = get_summpair_x_y_series(m1=m1, m2=m2, sd=sd, summpairs=summpairs, x_type=x_type, y_type=y_type)
            plot_X_Y(X, Y, x_type, y_type, m1, m2, abs_pr_plot_cutoff, fit_line_window)


# plotting graphs where points are summaries
def get_summ_x_y_series(m1, m2, sd, x_type, y_type):
    # data = []
    X = []
    Y = []
    for doc_id, isd in sd.items():
        ref_summ = isd['ref_summ']
        for sysname, sumdict in isd['system_summaries'].items():
            x_val = get_summ_x_val(ref_summ, m1, m2, sumdict, x_type)
            y_val = get_summ_y_val(ref_summ, m1, m2, sumdict, y_type)
            X.append(x_val)
            Y.append(y_val)
        print(f"get_summ_x_y_series: done {doc_id}/{len(sd)}", end="\r")
    return X, Y


def get_summ_x_val(ref, m1, m2, sumdict, x_type):
    assert x_type in summ_x_types
    if x_type == 'm_avg_norm':
        return (sumdict['normed_scores'][m1] + sumdict['normed_scores'][m2]) / 2
    elif x_type == 'm_avg':
        return (sumdict['scores'][m1] + sumdict['scores'][m2]) / 2
    elif x_type == 'm1':
        return sumdict['scores'][m1]
    elif x_type == 'm2':
        return sumdict['scores'][m2]
    elif x_type == 'm1_norm':
        return sumdict['normed_scores'][m1]
    elif x_type == 'm2_norm':
        return sumdict['normed_scores'][m2]


def get_summ_y_val(ref, m1, m2, sumdict, y_type):
    assert y_type in summ_y_types
    if y_type == 'normed_scores_abs_diff':
        return abs(sumdict['normed_scores'][m1] - sumdict['normed_scores'][m2])
    elif y_type == 'normed_scores_diff':
        return (sumdict['normed_scores'][m1] - sumdict['normed_scores'][m2])
    elif y_type == 'scores_abs_diff':
        return abs(sumdict['scores'][m1] - sumdict['scores'][m2])
    elif y_type == 'scores_diff':
        return (sumdict['scores'][m1] - sumdict['scores'][m2])


def plot_summ(sd, x_type, y_type, metrics_list=None, abs_pr_plot_cutoff=None, fit_line_window=100):
    if not metrics_list:
        metrics_list = get_metrics_list(sd)
    for m1_idx, m1 in enumerate(metrics_list):
        for m2 in metrics_list[m1_idx + 1:]:
            X, Y = get_summ_x_y_series(m1=m1, m2=m2, sd=sd, x_type=x_type, y_type=y_type)
            plot_X_Y(X, Y, x_type, y_type, m1, m2, abs_pr_plot_cutoff, fit_line_window)


# plotting graphs where points are documents
def get_doc_x_y_series(m1, m2, sd, x_type, y_type, x_modifier=None, m_list=None,
                       cutoff_metric=None, percentile=None, p_val_thresh=0.05):
    # data = []
    X = []
    Y = []
    num_ignored = 0
    for doc_id, isd in sd.items():
        x_val = get_doc_x_val(isd, m1, m2, x_type, x_modifier, m_list=m_list)
        y_val, p_val = get_doc_y_val(isd, m1, m2, y_type, cutoff_metric=cutoff_metric, percentile=percentile)
        if math.isnan(y_val) or p_val > p_val_thresh:
            num_ignored += 1
            continue
        X.append(x_val)
        Y.append(y_val)
        print(f"get_doc_x_y_series: done {doc_id}/{len(sd)}", end="\r")
    print(f"num ignored:{num_ignored}/{len(sd)}             ")
    return X, Y


def get_doc_x_val(isd, m1, m2, x_type, x_modifier, m_list=None):
    assert x_type in (doc_x_types + more_doc_x_types + ref_summ_x_types)
    if 'all' in x_type:
        assert m_list is not None  # we only need m_list if all all_metrics are being used in this x-value
    if x_type == 'avg__avg_m1_avg_m2':
        return np.mean([
            isd['mean_scores'][m1],
            isd['mean_scores'][m2]
        ])
    elif x_type == 'avg__all_avg_m':
        return np.mean([
            isd['mean_scores'][m] for m in m_list
        ])
    elif x_type == 'avg_m1':
        return isd['mean_scores'][m1]
    elif x_type == 'avg_m2':
        return isd['mean_scores'][m2]
    elif x_type == 'avg__max_m1_max_m2':
        return np.mean([
            isd['max_scores'][m1],
            isd['max_scores'][m2]
        ])
    elif x_type == 'avg__all_max_m':
        return np.mean([
            isd['max_scores'][m] for m in m_list
        ])
    elif x_type == 'max_m1':
        return isd['max_scores'][m1]
    elif x_type == 'max_m2':
        return isd['max_scores'][m2]
    elif x_type == 'compression':
        return isd['compression']
    elif x_type == 'coverage':
        return isd['coverage']
    elif x_type == 'density':
        return isd['density']
    else:
        return get_ref_summ_x_val(ref=isd['ref_summ'], summ='<t> dummmyy </t>', x_type=x_type,
                                  x_modifier=x_modifier, isd=isd)


def filter_summaries(isd, cutoff_metric, percentile):
    c_scores = [summdict['scores'][cutoff_metric] for summdict in isd['system_summaries'].values()]
    cutoff_score_min = np.percentile(c_scores, percentile[0])
    cutoff_score_max = np.percentile(c_scores, percentile[1])
    filtered_sumdicts_l = [summdict for summdict in isd['system_summaries'].values()
                           if (
                                   (summdict['scores'][cutoff_metric] >= cutoff_score_min)
                                   and (summdict['scores'][cutoff_metric] <= cutoff_score_max)
                           )
                           ]
    return filtered_sumdicts_l


def get_doc_y_val(isd, m1, m2, y_type, cutoff_metric=None, percentile=None):
    assert (y_type in doc_y_types)
    if y_type == 'm':
        return isd['mean_scores'][m1], 0

    filtered_summaries = isd['system_summaries'].values()
    if cutoff_metric is not None:
        filtered_summaries = filter_summaries(isd, cutoff_metric, percentile)

    m1_scores = [summdict['scores'][m1] for summdict in filtered_summaries]
    m2_scores = [summdict['scores'][m2] for summdict in filtered_summaries]

    if y_type == 'ktau':
        ktau, pval = kendalltau(m1_scores, m2_scores, nan_policy="raise")
        if np.isnan(ktau):
            # return high pvalue to ignore
            return 0, 1
            # return 0, 1, 1
            # import pdb; pdb.set_trace()
        return ktau, pval
        # return ktau, pval, len(filtered_summaries)

    elif y_type == 'pearson':
        pearson_corr, pval = pearsonr(m1_scores, m2_scores)
        return pearson_corr, pval

    elif y_type == 'spearman':
        # TODO: is this tested?
        spearman_corr, pval = spearmanr(m1_scores, m2_scores)
        return spearman_corr, pval


def plot_doc(sd, x_type, y_type, x_modifier=None, metrics_list=None, take_metric_pairs=True,
             abs_pr_plot_cutoff=None, fit_line_window=100,
             cutoff_metric=None, percentile=None, metric_pairs=None,
             show_y_label=True, show_title=True, show_yticklabels=True, save_fig=False):
    if not metrics_list:
        metrics_list = get_metrics_list(sd)
    if take_metric_pairs:
        if metric_pairs is None:
            m_pair_list = itertools.combinations(metrics_list, 2)
        else:
            m_pair_list = metric_pairs
    else:
        m_pair_list = [(m, '') for m in metrics_list]

    for m1, m2 in m_pair_list:
        X, Y = get_doc_x_y_series(m1=m1, m2=m2, sd=sd, x_type=x_type,
                                  x_modifier=x_modifier, y_type=y_type, m_list=metrics_list,
                                  cutoff_metric=cutoff_metric, percentile=percentile)
        plot_X_Y(X, Y, x_type, y_type, m1, m2, abs_pr_plot_cutoff, fit_line_window,
                 show_y_label=show_y_label, show_title=show_title, show_yticklabels=show_yticklabels, save_fig=save_fig)


def get_ref_summ_x_val(ref, summ, x_type, x_modifier, isd=None):
    assert x_type in ref_summ_x_types
    assert x_modifier in ref_summ_x_modifiers
    ref_sent_l = text_to_sent_l(ref)
    summ_sent_l = text_to_sent_l(summ)
    ref_words_l, ref_vocab_s = get_words_list_and_n_grams_set(ref, n=1, sent_tags=True)
    summ_words_l, summ_vocab_s = get_words_list_and_n_grams_set(summ, n=1, sent_tags=True)
    # ref_unique_2_grams': get_n_grams(sd[doc_idx]['ref_summ'], n=2)

    if ('doc' in x_type) or ('doc' in x_modifier):
        doc_sent_s = set()
        for d in isd['system_summaries'].values():
            this_sent_l = text_to_sent_l(d['system_summary'])
            doc_sent_s = doc_sent_s.union(set(this_sent_l))
        doc = tagify(list(doc_sent_s))
        doc_sent_l = text_to_sent_l(doc)
        doc_words_l, doc_vocab_s = get_words_list_and_n_grams_set(doc, n=1, sent_tags=True)
    # pdb.set_trace()
    if x_type == 'summ_abstractiveness_wrt_ref':
        overlap = 1 - (len(ref_vocab_s.intersection(summ_vocab_s)) / len(summ_vocab_s))
        return overlap
    elif x_type == 'ref_abstractiveness_wrt_doc':
        overlap = 1 - (len(ref_vocab_s.intersection(doc_vocab_s)) / len(ref_vocab_s))
        return overlap
    elif x_type == 'vocab_by_nwords':
        ref_val = len(ref_vocab_s) / len(ref_words_l)
        summ_val = len(summ_vocab_s) / len(summ_words_l)
        if doc_vocab_s:
            doc_val = len(doc_vocab_s) / len(doc_words_l)
    elif x_type == 'n_words':
        ref_val = len(ref_words_l)
        summ_val = len(summ_words_l)
        if doc_vocab_s:
            doc_val = len(doc_words_l)
    elif x_type == 'n_sents':
        ref_val = len(ref_sent_l)
        summ_val = len(summ_sent_l)
        if doc_vocab_s:
            doc_val = len(doc_sent_l)
    elif x_type == 'n_words_per_sent':
        ref_val = len(ref_words_l) / len(ref_sent_l)
        summ_val = len(summ_words_l) / len(summ_sent_l)
        if doc_vocab_s:
            doc_val = len(doc_words_l) / len(doc_sent_l)

    if x_modifier == 'ref':
        return ref_val
    elif x_modifier == 'summ':
        return summ_val
    elif x_modifier == 'doc':
        return doc_val
    elif x_modifier == 'ref_summ_diff':
        return ref_val - summ_val
    elif x_modifier == 'ref_summ_abs_diff':
        return abs(ref_val - summ_val)


def text_to_sent_l(text):
    sents = re.findall(r'%s (.+?) %s' % ('<t>', '</t>'), text)
    return sents


def tagify(l):
    return '<t> ' + ' </t> <t> '.join(l) + ' </t>'


def get_words_list_and_n_grams_set(text, n, sent_tags=True):
    n_grams = []
    text = text.lower()
    if sent_tags:
        sents = text_to_sent_l(text)
        words = [word for sent in sents for word in sent.split(' ')]
    else:
        words = text.split(' ')
    for i in range(0, len(words) - n):
        n_grams.append(tuple(words[i: i + n]))
    return words, set(n_grams)
