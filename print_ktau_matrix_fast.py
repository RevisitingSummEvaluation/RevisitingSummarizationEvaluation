import argparse
import numpy as np
from scipy.stats import kendalltau
from joblib import Parallel, delayed
from joblib import parallel_backend

import analysis_functions as af


def get_perc2score(sd, perc, equal_width=True):
    doc2perc = {}
    for doc_id, isd in sd.items():
        nas_scores = [summdict['scores']['nas'] for summdict in isd['system_summaries'].values()]
        perc_scores = {}
        if equal_width:
            max_score = np.max(nas_scores)
            for p in perc:
                perc_scores[p] = (p/100) * max_score
        else:
            for p in perc:
                perc_scores[p] = np.percentile(nas_scores, p)
        doc2perc[doc_id] = perc_scores
    return doc2perc


def get_ktau(doc_id, doc2perc, m1, m2, perc):
    isd = sd[doc_id]
    min_score = doc2perc[doc_id][perc[0]]
    max_score = doc2perc[doc_id][perc[1]]
    filtered_summaries = [summdict for summdict in isd['system_summaries'].values()
                           if (
                                   (summdict['scores']['nas'] >= min_score)
                                   and (summdict['scores']['nas'] <= max_score)
                           )
                           ]
    m1_scores = [summdict['scores'][m1] for summdict in filtered_summaries]
    m2_scores = [summdict['scores'][m2] for summdict in filtered_summaries]

    ktau, pval = kendalltau(m1_scores, m2_scores, nan_policy="raise")
    if np.isnan(ktau):
        return 0, 1
    return ktau, pval


def print_ktau_matrix(metrics, percentile, n_jobs=16):
    high_pvals = 0
    print(metrics)
    doc2perc = get_perc2score(sd, perc=(0, 33, 66, 100), equal_width=True)
    avg_perc = {p: np.mean([perc[p] for perc in doc2perc.values()]) for p in [0, 33, 66, 100]}
    print(avg_perc)
    for min_j, mx in enumerate(metrics):
        mean_ktaus = np.zeros((len(percentile), len(metrics)))
        for i, perc in enumerate(percentile):
            for j, my in enumerate(metrics):
                if mx == my or j <= min_j:
                    continue
                with parallel_backend('threading', n_jobs=n_jobs):
                    ktaus_pvals = Parallel()(
                        delayed(get_ktau)(doc_id, doc2perc, mx, my, perc)
                        for doc_id, isd in sd.items()
                    )
                ktaus = []
                for ktau, pval in ktaus_pvals:
                    if pval <= 0.05:
                        ktaus.append(ktau)
                    else:
                        high_pvals += 1
                mean_ktaus[i, j] = np.mean(ktaus)

        print(mean_ktaus)
        print()

    total_ktaus = (len(metrics) / 2) * (len(metrics) - 1) * len(percentile) * len(sd)
    print(f"total {high_pvals}/{total_ktaus} = {high_pvals * 100 / total_ktaus}% values ignored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', required=True, help='path to sd')
    args = parser.parse_args()

    sd = af.get_pickle(args.sd)
    mlist = ['bert_recall_score', 'mover_score', 'rouge_1_recall', 'rouge_2_recall', 'rouge_l_recall', 'js-2']
    for isd in sd.values():
        for sys_name in isd['system_summaries']:
            scores = isd['system_summaries'][sys_name]['scores']
            scores['nas'] = np.mean([isd['system_summaries'][sys_name]['normed_scores'][m] for m in mlist])
    metrics = mlist
    print_ktau_matrix(metrics, percentile=[(0, 100), (33, 100), (66, 100)])
    print_ktau_matrix(metrics, percentile=[(0, 33), (33, 66), (66, 100)])
