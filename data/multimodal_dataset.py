import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from data.base_dataset import BaseDataset


class MultimodalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='IEMOCAP', help='which dataset to use')
        return parser
    
    def __init__(self, opt, set_name):
        ''' IEMOCAP dataset reader
            set_name in ['trn', 'val', 'tst']
        '''
        super().__init__(opt)

        # record & load basic settings 
        cvNo = opt.cvNo
        self.set_name = set_name
        pwd = os.path.abspath(__file__)
        pwd = os.path.dirname(pwd)
        config = json.load(open(os.path.join(pwd, 'config', f'{opt.corpus_name}_config.json')))
        self.norm_method = opt.norm_method
        self.corpus_name = opt.corpus_name
        # load feature
        self.A_type = opt.A_type
        self.all_A = \
            h5py.File(os.path.join(config['feature_root'], 'A', f'{self.A_type}.h5'), 'r')
        if self.A_type == 'comparE':
            self.mean_std = h5py.File(os.path.join(config['feature_root'], 'A', 'comparE_mean_std.h5'), 'r')
            self.mean = torch.from_numpy(self.mean_std[str(cvNo)]['mean'][()]).unsqueeze(0).float()
            self.std = torch.from_numpy(self.mean_std[str(cvNo)]['std'][()]).unsqueeze(0).float()
        elif self.A_type == 'comparE_raw':
            self.mean, self.std = self.calc_mean_std()
            
        self.V_type = opt.V_type
        self.all_V = \
            h5py.File(os.path.join(config['feature_root'], 'V', f'{self.V_type}.h5'), 'r')
        self.L_type = opt.L_type
        self.all_L = \
            h5py.File(os.path.join(config['feature_root'], 'L', f'{self.L_type}.h5'), 'r')
            
        # load dataset in memory
        if opt.in_mem:
            self.all_A = self.h5_to_dict(self.all_A)
            self.all_V = self.h5_to_dict(self.all_V)
            self.all_L = self.h5_to_dict(self.all_L)
        
        # load target
        label_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_label.npy")
        int2name_path = os.path.join(config['target_root'], f'{cvNo}', f"{set_name}_int2name.npy")
        self.label = np.load(label_path)
        if self.corpus_name == 'IEMOCAP':
            self.label = np.argmax(self.label, axis=1)
        self.int2name = np.load(int2name_path)
        self.manual_collate_fn = True

    def __getitem__(self, index):
        int2name = self.int2name[index]
        if self.corpus_name == 'IEMOCAP':
            int2name = int2name[0].decode()
        label = torch.tensor(self.label[index])
        # process A_feat
        A_feat = torch.from_numpy(self.all_A[int2name][()]).float()
        if self.A_type == 'comparE' or self.A_type == 'comparE_raw':
            A_feat = self.normalize_on_utt(A_feat) if self.norm_method == 'utt' else self.normalize_on_trn(A_feat)
        # process V_feat 
        V_feat = torch.from_numpy(self.all_V[int2name][()]).float()
        # process L_feat
        L_feat = torch.from_numpy(self.all_L[int2name][()]).float()
        return {
            'A_feat': A_feat, 
            'V_feat': V_feat,
            'L_feat': L_feat,
            'label': label,
            'int2name': int2name
        }
    
    def __len__(self):
        return len(self.label)

    def h5_to_dict(self, h5f):
        ret = {}
        for key in h5f.keys():
            ret[key] = h5f[key][()]
        return ret
    
    def normalize_on_utt(self, features):
        mean_f = torch.mean(features, dim=0).unsqueeze(0).float()
        std_f = torch.std(features, dim=0).unsqueeze(0).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    def normalize_on_trn(self, features):
        features = (features - self.mean) / self.std
        return features

    def calc_mean_std(self):
        utt_ids = [utt_id for utt_id in self.all_A.keys()]
        feats = np.array([self.all_A[utt_id] for utt_id in utt_ids])
        _feats = feats.reshape(-1, feats.shape[2])
        mean = np.mean(_feats, axis=0)
        std = np.std(_feats, axis=0)
        std[std == 0.0] = 1.0
        return mean, std

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        int2name = [sample['int2name'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'lengths': lengths,
            'int2name': int2name
        }


if __name__ == '__main__':
    class test:
        cvNo = 1
        A_type = "comparE"
        V_type = "denseface"
        L_type = "bert_large"
        norm_method = 'trn'

    
    opt = test()
    print('Reading from dataset:')
    a = MultimodalDataset(opt, set_name='trn')
    data = next(iter(a))
    for k, v in data.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    print('Reading from dataloader:')
    x = [a[100], a[34], a[890]]
    print('each one:')
    for i, _x in enumerate(x):
        print(i, ':')
        for k, v in _x.items():
            if k not in ['int2name', 'label']:
                print(k, v.shape)
            else:
                print(k, v)
    print('packed output')
    x = a.collate_fn(x)
    for k, v in x.items():
        if k not in ['int2name', 'label']:
            print(k, v.shape)
        else:
            print(k, v)
    