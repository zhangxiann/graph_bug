# -- coding: utf-8 --**
# the dataset class for EV-Gait-3DGraph model


import os
import time

import numpy as np
import glob
import scipy.io as sio
import torch
import torch.utils.data
from torch._six import container_abcs
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Dataset, InMemoryDataset, Batch
import os.path as osp


def files_exist(files):
    return all([osp.exists(f) for f in files])

class Graph_2D_Memory_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Graph_2D_Memory_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # return file list of self.raw_dir
    @property
    def raw_file_names(self):
        all_filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        # get all file names
        file_names = [f.split(os.sep)[-1] for f in all_filenames]
        return file_names

    # get all file names in  self.processed_dir
    @property
    def processed_file_names(self):
        return ['data.pt']


    def download(self):
        pass


    # convert the mat files of self.raw_dir to torch_geometric.Data format, save the result files in self.processed_dir
    # this method will only execute one time at the first running.
    def process(self):
        data_list = []
        for raw_path in self.raw_paths:

            content = sio.loadmat(raw_path)
            feature = torch.tensor(content["feature"]).float()
            edge_index = torch.tensor(
                np.array(content["edges"]).astype(np.int32), dtype=torch.long
            )
            # 构建 2D Graph
            pos = torch.tensor(content["pseudo"], dtype=torch.float32)[:, 1:3]
            # pos = torch.tensor(np.array(content["pseudo"]), dtype=torch.float32)
            label_idx = torch.tensor(int(content["label"]), dtype=torch.long)
            data = Data(
                x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0)
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])