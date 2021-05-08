# -- coding: utf-8 --**
# train the Run EV-Gait-3DGraph model

# nohup python -u EV-Gait-3DGraph/train_3d_graph_gait.py --type "train 2020_zx_outdoor_day1 test 2020_zx_outdoor_day2" --cuda 1 --experiment_nums 1 --result_file "gcn_result_zx_cv.log" > gcn_cv_zx.log 2>&1 &
import time

import numpy as np


import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import argparse
from graph_2d_memory_dataset import Graph_2D_Memory_Dataset
from tqdm import tqdm

from model_2dgraph import *
import os
import logging


from graph_2d_dataset import *



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=10, type=int, help="The GPU ID")
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument("--workers", default=0, type=int, help="number of workers")




    args = parser.parse_args()


    train_dir = 'data/graph/Train'
    test_dir = 'data/graph/Test'


    n_class=10
    model = Net_n_mnist()
    model_dir = 'n_mnist_model'
    log_dir ='n_mnist_log'

    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logging.basicConfig(filename=os.path.join(log_dir, 'n_mnist.log'), level=logging.DEBUG)
    model_file = 'n_mnist.pkl'

    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    pre_transform = T.Compose([T.Cartesian(cat=False)])
    # train_data_aug = T.Compose([T.RandomScale([0.95, 1])])
    train_data_aug = T.Compose([T.RandomScale([0.95, 1]), T.RandomRotate((0 ,10), axis=0), T.RandomFlip(axis=0, p=0.5)])

    train_dataset = Graph_2D_Memory_Dataset(
        train_dir, transform=train_data_aug, pre_transform = pre_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # train
    print("train")
    for epoch in range(1, args.epoch):
        model.train()
        correct = 0
        total = 0
        time2 = time.time()

        for i, data in enumerate(tqdm(train_loader)):
            # time1 = time.time()
            # print("read data: {}".format(time1-time2))
            data = data.to(device)
            optimizer.zero_grad()
            end_point = model(data)
            loss = F.nll_loss(end_point, data.y)
            pred = end_point.max(1)[1]
            total += len(data.y)
            correct += pred.eq(data.y).sum().item()
            loss.backward()
            optimizer.step()
            # time2 = time.time()
            # print("model forward: {}".format(time2 - time1))

        # accuracy of each epoch
        logging.info("epoch: {}, train acc is {}".format(epoch, float(correct) / total))
        print("epoch: {}, train acc is {}".format(epoch, float(correct) / total))



