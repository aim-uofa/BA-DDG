import os
import copy
import math
import random
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import json
import torch
from torch.utils.data import Dataset

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Polypeptide import one_to_index, index_to_one
from common_utils.transforms import get_transform
from common_utils.protein.parsers import parse_biopython_structure
from torch.utils.data._utils.collate import default_collate
from copy import deepcopy


class SkempiDataset(Dataset):

    def __init__(self, csv_path, pdb_dir, cache_dir, num_cvfolds=3, cvfold_index=0, split='train', reset=False):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.num_cvfolds = num_cvfolds
        self.cvfold_index = cvfold_index
        self.transform = get_transform([{'type': 'select_atom', 'resolution': 'backbone+CB'},  {'type': 'corrupt_chi_angle', 'ratio_mask': 0.1}])
        self.split = split

        self.entries_cache = os.path.join(cache_dir, 'entries.pkl')
            
        self.entries = None
        self.entries_full = None

        self._load_entries(reset)
        
        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
        self.structures = None
        self._load_structures(reset)
    
    
    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        
        random.Random(3745754758).shuffle(complex_list) # fixed seed for reproducibility, same as Prompt-DDG
        split_size = math.ceil(len(complex_list) / self.num_cvfolds)
        complex_splits = [
            complex_list[i*split_size : (i+1)*split_size] 
            for i in range(self.num_cvfolds)
        ]

        val_split = complex_splits.pop(self.cvfold_index)
        train_split = sum(complex_splits, start=[]) if self.num_cvfolds > 1 else val_split
        if self.num_cvfolds == 1: 
            val_split = ['3VR6_ABCDEF_GH'] 
        if '3VR6_ABCDEF_GH' in train_split: # remove 3VR6_ABCDEF_GH from train split, due to limit of GPU memory
            train_split.remove('3VR6_ABCDEF_GH')
        
        if self.split == 'val':
            complexes_this = val_split
        elif self.split == 'all':
            complexes_this = complex_list
        else:
            complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            entries += complex_to_entries[cplx]
        
        self.entries = entries
        
    def _preprocess_entries(self):
        entries = load_skempi_entries(self.csv_path, self.pdb_dir)
        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.parser = PDBParser(QUIET=True)
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}
        complex = list(set([e['complex'] for e in self.entries_full]))
        for complex in tqdm(complex, desc='Structures'):
            pdbcode = complex.split('_')[0]                    
            antibody_chain_id = list(complex.split('_')[-1])
            antigen_chain_id = list(complex.split('_')[-2])
            pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))
            model = self.parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model, antibody_chain_id=antibody_chain_id, antigen_chain_id=antigen_chain_id)
            structures[pdbcode] = (data, seq_map)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
        data['ddG'] = np.float32(entry['ddG'])
        data['dG_wt'] = np.float32(entry['dG_wt'])
        data['dG_mut'] = np.float32(entry['dG_mut'])
        data['id'] = entry['id']
        data['temperature'] = np.float32(entry['temperature'])
        data['complex'] = entry['complex']
        data['num_muts'] = entry['num_muts']

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'])
            if ch_rs_ic not in seq_map: continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
            
        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])
        data['mutstr'] = ','.join('{}{}{}{}'.format(
            mut['wt'],
            mut['chain'],
            mut['resseq'],
            mut['mt']
        ) for mut in entry['mutations'])

        if self.transform is not None:
            data = self.transform(data)

        return data

import re
def process_temperature(temp):
    match = re.search(r'\d+', str(temp))
    return int(match.group()) if match else None

def load_skempi_entries(csv_path, pdb_dir, block_list={'1KBH'}):
    df = pd.read_csv(csv_path, sep=';')
    
    df['temperature'] = df['Temperature'].apply(process_temperature)
    df.fillna({'temperature': 25.0 + 273.15}, inplace=True)
    df['dG_wt'] =  (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] =  (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']

    def _parse_mut(mut_name):
        return {'wt': mut_name[0], 'mt': mut_name[-1], 'chain': mut_name[1], 'resseq': int(mut_name[2:-1])}

    entries = []
    for i, row in df.iterrows():
        pdbcode, _, _ = row['#Pdb'].split('_')

        if pdbcode in block_list:
            continue
        if not os.path.exists(os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))):
            continue
        if not np.isfinite(row['ddG']):
            continue

        mutations = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        entry = {
            'id': i,
            'complex': row['#Pdb'],
            'pdbcode': pdbcode,
            'mutations': mutations,
            'num_muts': len(mutations),
            'ddG': np.float32(row['ddG']),
            'dG_wt': np.float32(row['dG_wt']),
            'dG_mut': np.float32(row['dG_mut']),
            'temperature': np.float32(row['temperature']),
            'mutstr': row['Mutation(s)_cleaned'],
        }
        entries.append(entry)

    return entries

def reset_residue_idx(res_nb):
    reset_points = (res_nb == 1).nonzero(as_tuple=True)[0][1:]
    offsets = torch.zeros_like(res_nb)
    offsets[reset_points] = 100 + res_nb[reset_points-1]
    offsets = torch.cumsum(offsets, dim=0)
    return res_nb + offsets
