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

class Graph_2D_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Graph_2D_Dataset, self).__init__(root, transform, pre_transform)
        self.datas=[]
        if files_exist(self.processed_paths):
            print("preload data")
            self.datas = [torch.load(path) for path in self.processed_paths]
        print("data size = {}".format(len(self.datas)))

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
        filenames = glob.glob(os.path.join(self.raw_dir, "*.mat"))
        file = [f.split(os.sep)[-1] for f in filenames]
        saved_file = [f.replace(".mat", ".pt") for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass


    # convert the mat files of self.raw_dir to torch_geometric.Data format, save the result files in self.processed_dir
    # this method will only execute one time at the first running.
    def process(self):
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

            saved_name = raw_path.split(os.sep)[-1].replace(".mat", ".pt")
            torch.save(data, osp.join(self.processed_dir, saved_name))
        self.datas = [torch.load(path) for path in self.processed_paths]

    def get(self, idx):
        # data = torch.load(osp.join(self.processed_paths[idx]))
        # return data
        return self.datas[idx]


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

# class MyCollater(object):
#     def __init__(self, follow_batch):
#         self.follow_batch = follow_batch
#
#     def collate(self, batch):
#
#         elem = batch[0]
#         if isinstance(elem, Data):
#             return Batch.from_data_list(batch, self.follow_batch)
#         elif isinstance(elem, torch.Tensor):
#             return default_collate(batch)
#         elif isinstance(elem, float):
#             return torch.tensor(batch, dtype=torch.float)
#         elif isinstance(elem, int):
#             return torch.tensor(batch)
#         elif isinstance(elem, (str, bytes)):
#             return batch
#         elif isinstance(elem, container_abcs.Mapping):
#             return {key: self.collate([d[key] for d in batch]) for key in elem}
#         elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
#             return type(elem)(*(self.collate(s) for s in zip(*batch)))
#         elif isinstance(elem, container_abcs.Sequence):
#             return [self.collate(s) for s in zip(*batch)]
#
#
#         raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))
#
#     def __call__(self, batch):
#         time1 = time.time()
#         res = self.collate(batch)
#         time2 = time.time()
#         print("collate time = {}".format(time2-time1))
#         return res
#
# class DataLoaderX(torch.utils.data.DataLoader):
#     r"""Data loader which merges data objects from a
#     :class:`torch_geometric.data.dataset` to a mini-batch.
#
#     Args:
#         dataset (Dataset): The dataset from which to load the data.
#         batch_size (int, optional): How many samples per batch to load.
#             (default: :obj:`1`)
#         shuffle (bool, optional): If set to :obj:`True`, the data will be
#             reshuffled at every epoch. (default: :obj:`False`)
#         follow_batch (list or tuple, optional): Creates assignment batch
#             vectors for each key in the list. (default: :obj:`[]`)
#     """
#
#     def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
#                  **kwargs):
#         super(DataLoaderX,
#               self).__init__(dataset, batch_size, shuffle,
#                              collate_fn=MyCollater(follow_batch), **kwargs)