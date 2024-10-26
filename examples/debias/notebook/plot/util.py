from pymongo import MongoClient
from pathlib import Path
import yaml
import os
import sys
import numpy as np
import matplotlib_inline.backend_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('pdf')
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../../') # Hack!
from src.util.parse import parse_n_expr, parse_n_m_dict

def load_db(collection_name, *,
            method_record_dict,
            ns, n_key,
            seeds, seed_key,
            metrics):
    db_yaml = '../../src/config/cache/mongodb.yaml'
    db_info = yaml.safe_load(open(db_yaml,'r').read())

    client = MongoClient(db_info['uri'])
    collection = client[db_info['db_name']][collection_name]

    methods = method_record_dict.keys()
    loaded_result = {}
    for method in methods:
        record = method_record_dict[method]
        result_med = np.zeros((len(ns), len(metrics)))
        result_std = np.zeros_like(result_med)
        result_m = np.zeros((len(ns),))
        for i, n in enumerate(ns):
            results = {metric: [] for metric in metrics}
            record_cp = record.copy()
            # Parsing out size
            del record_cp['out_size']
            if record['compress.alg'] == 'cpp':
                m = parse_n_expr('n**0.5', n) # always m = sqrt(n)
            else:
                m = parse_n_expr(record['out_size'], n)
            if record['compress.alg'] != 'cpp':
                record_cp['compress.out_size'] = m
            record_cp = parse_n_m_dict(record_cp, n, m,
                                       keywords=['debias.rank'])
            result_m[i] = m
            for seed in seeds:
                filter = {
                        '_stage': 'final',
                        n_key: n,
                        seed_key: seed,
                        **record_cp,
                }
                try:
                    count = collection.count_documents(filter)
                    if (count != 1):
                        print(f'Found {count} matches for record = {filter}!')
                        if count > 1:
                            cursor = collection.find(filter)
                            for t_doc in cursor:
                                print(t_doc)
                            raise Exception('Record not unique!')
                        else:
                            raise Exception('Record not found!')
                    doc = collection.find_one(filter)

                    for metric in metrics:
                        results[metric].append(doc['_result'][metric])
                except Exception as e:
                    print(e)
            if len(results[metrics[0]]) == 0:
                result_med[i, :] = np.nan
                result_std[i, :] = np.nan
            else:
                for l, metric in enumerate(metrics):
                    arr = np.array(results[metric])
                    result_med[i, l] = np.median(arr)
                    result_std[i, l] = arr.std()
        loaded_result[method] = {
            'median': result_med,
            'std': result_std,
            'ms': result_m,
        }
    return loaded_result


from style import *
def plot_one_col(loaded_results, metrics, titles,
                 use_tex=True, figsize=5, log_x_axis=False,
                 aspect=1.5, save_path=None,
                 metric_y_lim=None, x_lim=None,
                 log_lim=True, include_y_label=True, share_y=False):

    if isinstance(loaded_results, dict):
        loaded_results = [loaded_results]
        titles = [titles]

    if use_tex:
        plt.rcParams['text.usetex'] = True
    plt.rcParams.update({
        'font.size': '18'
    })
    num_metric = len(metrics)
    num_col = len(loaded_results)
    fig, axes = plt.subplots(num_metric, num_col, figsize=(aspect*figsize*num_col,
                                                     figsize*num_metric),
                            squeeze=False,
                            sharey=share_y)
    for i, loaded_result in enumerate(loaded_results):
        for l, metric in enumerate(metrics):
            ax = axes[l, i]
            for method, result in loaded_result.items():
                marker = method_to_marker.get(method, 'X')
                ms = result['ms']
                ax.plot(ms, result['median'][:, l],
                        label=(method_to_tex.get(method, method)),
                        marker=marker,
                        markersize=18,
                        linestyle=method_to_ls.get(method, 'solid'),
                        linewidth=3,
                       color=method_to_color[method])
            if metric_y_lim is not None and metric in metric_y_lim:
                ax.set_ylim(*metric_y_lim[metric])
            if x_lim is not None:
                ax.set_xlim(*x_lim)
            if include_y_label:
                ax.set_ylabel(metric_to_tex[metric], fontsize=28)
            if l == 0:
                ax.set_title(titles[i], fontsize=28)
                plt.setp(ax.get_xticklabels(), visible=False)
            if l == len(metrics) - 1:
                ax.set_xlabel(f'Coreset Size $m$', fontsize=28)
                ax.legend()
            if metric != 'final_ed':
                ax.set_yscale('log')
            if log_x_axis:
                ax.set_xscale('log')
            if log_lim:
                print(f'xlim for {metric}: {ax.get_xlim()}')
                print(f'ylim for {metric}: {ax.get_ylim()}')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
