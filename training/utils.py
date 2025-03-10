import os
import yaml
import shutil
import random
import numpy as np
import pandas as pd

import torch
import torch.linalg

from tqdm.auto import tqdm
from easydict import EasyDict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def check_dir(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def save_code(save_dir):
    save_code_dir = os.path.join(save_dir, 'codes/')
    check_dir(save_code_dir + 'common_utils/modules/')
    check_dir(save_code_dir + 'common_utils/protein/')
    check_dir(save_code_dir + 'common_utils/transforms/')

    for file in os.listdir("../code"):
        if '.py' in file:
            shutil.copyfile('../code/' + file, save_code_dir + file)
    for file in os.listdir("../code/common_utils/modules/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/modules/' + file, save_code_dir + 'common_utils/modules/' + file)
    for file in os.listdir("../code/common_utils/protein/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/protein/' + file, save_code_dir + 'common_utils/protein/' + file)
    for file in os.listdir("../code/common_utils/transforms/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/transforms/' + file, save_code_dir + 'common_utils/transforms/' + file)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    return config, config_name


def per_complex_corr(df, pred_attr='ddG_pred', limit=10):
    corr_table = []

    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')

        if len(df_cplx) < limit: 
            continue

        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', pred_attr]].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', pred_attr]].corr('spearman').iloc[0,1],
        })

    corr_table = pd.DataFrame(corr_table)
    avg = corr_table[['pearson', 'spearman']].mean()

    return avg['pearson'] , avg['spearman']


def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1]
    return {
        'overall_pearson': pearson, 
        'overall_spearman': spearman,
    }


def percomplex_correlations(df, return_details=False):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < 10: 
            continue
        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
        })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman']].mean()
    out = {
        'percomplex_pearson': average['pearson'],
        'percomplex_spearman': average['spearman'],
    }
    if return_details:
        return out, corr_table
    else:
        return out


def overall_auroc(df):
    score = roc_auc_score(
        (df['ddG'] > 0).to_numpy(),
        df['ddG_pred'].to_numpy()
    )
    return {
        'auroc': score,
    }


def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt( ((true - pred_corrected) ** 2).mean() )
    mae = np.abs(true - pred_corrected).mean()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def analyze_all_results(df):
    methods = df['method'].unique()
    funcs = [
        overall_correlations,
        overall_rmse_mae,
        overall_auroc,
        percomplex_correlations,
    ]
    analysis = []
    for method in methods:
        df_this = df[df['method'] == method]
        result = {
            'method': method,
        }
        for f in funcs:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis


def analyze_all_percomplex_correlations(df):
    methods = df['method'].unique()
    df_corr = []
    for method in methods:
        df_this = df[df['method'] == method]
        _, df_corr_this = percomplex_correlations(df_this, return_details=True)
        df_corr_this['method'] = method
        df_corr.append(df_corr_this)
    df_corr = pd.concat(df_corr).reset_index()
    return df_corr


def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_corr = analyze_all_percomplex_correlations(df_items)
    df_metrics['mode'] = mode
    return df_metrics


def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics
