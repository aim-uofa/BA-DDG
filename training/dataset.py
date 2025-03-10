import math
import functools
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from skempi import SkempiDataset
from copy import deepcopy

DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_nb': -1, 
    'chain_id': ' ', 
}


class PaddingCollate(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n, f'{x.size(0)} > {n}, {x}'
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        max_length = math.ceil(max_length / 8) * 8
        keys = self._get_common_keys(data_list)

        data_list_padded = []
        for data in data_list:
            data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in data.items() if k in keys}
            data_list_padded.append(data_padded)

        batch = default_collate(data_list_padded)
        return batch

MPNN_PAD_VALUES = {
    'S': 0, 
    'S_mut': 0,
    'chain_encoding_all': 0, 
    'residue_idx': -100,
    'mask': 0,
    'chain_M':0,
    'ref_log_probs': 0.0,
}


def reset_residue_idx(res_nb):
    reset_points = (res_nb == 1).nonzero(as_tuple=True)[0][1:]
    offsets = torch.zeros_like(res_nb)
    offsets[reset_points] = 100 + res_nb[reset_points-1]
    offsets = torch.cumsum(offsets, dim=0)
    return res_nb + offsets

class MPNNPaddingCollate(PaddingCollate):

    def __init__(self, length_ref_key='aa', pad_values=MPNN_PAD_VALUES):
        super().__init__(length_ref_key=length_ref_key, pad_values=pad_values)
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        max_length = math.ceil(max_length / 8) * 8

        data_list_padded = []
        num_mut_chains = []
        for data in data_list:
            mpnn_data = {
                'X': data['pos_atoms'][...,:4,:],
                'aa': data['aa'], 
                'aa_mut': data['aa_mut'],
                'mask': torch.ones_like(data['aa']),
                'chain_M': data['mut_flag'],
                'chain_encoding_all': data['chain_nb']+1,
                'residue_idx': reset_residue_idx(data['res_nb']),
                'complex': data['complex'],
                'ddG': data['ddG'],
                'id': data['id'],
                'num_muts': data['num_muts'],
                'mutstr': data['mutstr'],
            }
            mpnn_data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in mpnn_data.items()}
            data_list_padded.append(mpnn_data_padded)
            
            mut_indices = torch.nonzero(data['mut_flag'], as_tuple=True)[0]
            mut_chains = torch.unique(mpnn_data['chain_encoding_all'][mut_indices], dim=0)
            num_mut_chains.append(mut_chains.size(0))
            
            for chain_idx in mut_chains.tolist():
                chain_mask = mpnn_data_padded['chain_encoding_all'] == chain_idx
                single_chain_mpnn_data_padded = deepcopy(mpnn_data_padded)
                single_chain_mpnn_data_padded['mask'] = chain_mask
                single_chain_mpnn_data_padded['chain_M'] = chain_mask * mpnn_data_padded['chain_M']
                single_chain_mpnn_data_padded['X'] = single_chain_mpnn_data_padded['X'] * chain_mask[:,None,None]
                single_chain_mpnn_data_padded['aa'] = single_chain_mpnn_data_padded['aa'] * chain_mask
                single_chain_mpnn_data_padded['aa_mut'] = single_chain_mpnn_data_padded['aa_mut'] * chain_mask
                data_list_padded.append(single_chain_mpnn_data_padded)
        batch = default_collate(data_list_padded)
        batch['num_mut_chains'] = num_mut_chains
        
        return batch



def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class SkempiDatasetManager(object):
    def __init__(self, config, num_cvfolds, num_workers=4):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.num_workers = num_workers
        
        self.train_loaders = []
        self.val_loaders = []
        
        for fold in range(num_cvfolds):
            train_loader, val_loader, pretrain_loader = self.init_loaders(fold)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
        self.pretrain_loader = pretrain_loader

    def init_loaders(self, fold):
        
        gt_dataset = functools.partial(
            SkempiDataset,
            csv_path = "../data/SKEMPI_v2/skempi_v2.csv",
            pdb_dir = "../data/SKEMPI_v2/PDBs",
            cache_dir = "../data/SKEMPI_v2_cache",
            num_cvfolds = self.num_cvfolds,
            cvfold_index = fold,
        )  
        
        train_dataset = gt_dataset(split='train')
        val_dataset = gt_dataset(split='val')
        pretrain_dataset = gt_dataset(split='all')
        
        train_cplx = set([e['complex'] for e in train_dataset.entries])
        val_cplx = set([e['complex'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0 or self.num_cvfolds == 1, f'data leakage {leakage}'

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=MPNNPaddingCollate(), 
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=MPNNPaddingCollate(), 
            num_workers=self.num_workers
        )
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=MPNNPaddingCollate(),  
            num_workers=self.num_workers
        )
        pretrain_iterator = inf_iterator(pretrain_loader)
        self.all_loader = pretrain_loader

        print('Fold %d: Train %d, Val %d, All %d' % (fold, len(train_dataset), len(val_dataset), len(pretrain_dataset)))
        return train_iterator, val_loader, pretrain_iterator

    def get_train_loader(self, fold):
        return self.train_loaders[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]
    
