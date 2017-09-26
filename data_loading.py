from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

relative_path = '../../DL4Bio/feat66_tensorflow/'
trainList_addr = 'data/trainList'
validList_addr = 'data/validList'
testList_addr = 'data/testList'

class Protein_feat66_Dataset(Dataset):
    def __init__(self, relative_path, datalist_addr, batch_size=64,
                 max_seq_length=300, feature_size=66):
        self.relative_path = relative_path
        self.protein_list = self.read_list(relative_path + datalist_addr)
        self.max_seq_length = max_seq_length
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.dict_ss = {key: value for (key, value) in \
            zip(['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T'], range(8))}

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        protein_name = self.protein_list[idx]
        features, labels, masks, seq_len = self.read_protein(protein_name)
        return protein_name, torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(masks), seq_len


    def read_list(self, filename):
        """Given the filename storing all protein names, return a list of protein names.
        """
        proteins_names = []
        with open(filename) as f:
            for line in f:
                protein_name = line.rstrip('\n')
                protein_addr = relative_path + 'data/66FEAT/' + protein_name + '.66feat'
                proteins_names.append(protein_name)
        return proteins_names

    def read_protein(self, protein_name):
        """Given a protein name, return ndarrays of its features, secondary structure labels,
        masks and according sequence length.
        
        The returned ndarrays will be zero-padded or cutoff and choose specified 
        number of features.
        """
        protein_addr = self.relative_path + 'data/66FEAT/' + protein_name + '.66feat'
        ss_addr = self.relative_path + 'data/Angles/' + protein_name + '.ang'

        protein_features = np.zeros((self.max_seq_length, self.feature_size), dtype=np.float32)
        protein_labels = np.zeros(self.max_seq_length, dtype=np.int32)
        protein_masks = np.zeros(self.max_seq_length, np.float32)

        _protein_features = np.loadtxt(protein_addr)
        min_idx = min(self.max_seq_length, _protein_features.shape[0])
        protein_features[:min_idx, :self.feature_size] = _protein_features[:min_idx, :self.feature_size]

        _protein_labels = []
        with open(ss_addr) as f:
            next(f)
            for i, line in enumerate(f):
                line = line.split('\t')
                if line[0] == '0':
                    _protein_labels.append(self.dict_ss[line[3]])

        _protein_labels = np.array(_protein_labels).transpose()
        protein_labels[:min_idx] = _protein_labels[:min_idx]
        protein_masks[:min_idx] = 1.0

        seq_len = min_idx

        return protein_features, protein_labels, protein_masks, seq_len

if __name__ == '__main__':
    protein_dataset = Protein_feat66_Dataset(relative_path, trainList_addr)
    # print(len(protein_dataset))
    # for i in range(1):
    #     sample = protein_dataset[i]
    #     print(sample['name'])
    #     print(sample['features'])
    #     print(sample['labels'])

    dataloader = DataLoader(protein_dataset, batch_size=1,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        name, features, labels, masks, seq_len = sample_batched
        print(name)
        # print(features.size())
        print(labels)
        print(masks)
        print(seq_len)
        break