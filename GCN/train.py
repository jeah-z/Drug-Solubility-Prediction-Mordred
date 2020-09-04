# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch import SchNetModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher


def dataset_split(file):
    delaney = pd.read_csv("delaney.csv")
    test_set = delaney.sample(frac=0.1, random_state=0)
    train_set = delaney.drop(test_set.index)
    test_set.to_csv("delaney_test.csv", index=False)
    train_set.to_csv("delaney_train.csv", index=False)


def train(model="sch", epochs=80, device=th.device("cpu"), dataset=''):
    print("start")
    train_dir = "./"
    train_file = "train_smi.csv"
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_dataset.mode = "Train"
    alchemy_dataset.transform = None
    alchemy_dataset.file_path = train_file
    alchemy_dataset._load()

    test_dataset = TencentAlchemyDataset()
    test_dir = train_dir
    test_file = "val_smi.csv"
    test_dataset.mode = "Train"
    test_dataset.transform = None
    test_dataset.file_path = test_file
    test_dataset._load()

    alchemy_loader = DataLoader(
        dataset=alchemy_dataset,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=0,
    )

    if model == "sch":
        model = SchNetModel(norm=False, output_dim=1)
    elif model == "mgcn":
        model = MGCNModel(norm=False, output_dim=1)
    elif model == "MPNN":
        model = MPNNModel(output_dim=1)
    print(model)
    # if model.name in ["MGCN", "SchNet"]:
    #     model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)
    # print("test_dataset.mean= %s" % (alchemy_dataset.mean))
    # print("test_dataset.std= %s" % (alchemy_dataset.std))

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        w_mae /= idx + 1
        w_loss /= idx + 1

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))

        val_loss, val_mae = 0, 0
        if(epoch%50 == 0):
                res_file = open('val_results_%s.txt'%(epoch), 'w')
        for jdx, batch in enumerate(test_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            mae.backward()
            optimizer.step()

            val_mae += mae.detach().item()
            val_loss += loss.detach().item()

            res_np = res.cpu().detach().numpy()
            label_np = batch.label.cpu().detach().numpy()

            if(epoch%50 == 0):
                for i in range(len(res_np)):
                    res_file.write(str(res_np[i][0]) + '\t')
                    res_file.write(str(label_np[i][0])+'\n')

        val_mae /= jdx + 1
        val_loss /= jdx + 1
        print(
            "Epoch {:2d}, val_loss: {:.7f}, val_mae: {:.7f}".format(
                epoch, val_loss, val_mae
            ))

        if epoch % 50 == 0:
            th.save(model.state_dict(), './'+dataset+"/model_"+str(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch, mgcn or MPNN)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument("--dataset", help="dataset to train", default="")
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ["sch", "mgcn", "MPNN"]
    # dataset_split("delaney.csv")
    train(args.model, int(args.epochs), device, args.dataset)
