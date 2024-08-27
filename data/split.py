import argparse
import yaml
import os
import json
import logging
import torch
import pytorch_lightning as pl
from tqdm.autonotebook import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split
from experiments.downstream_model.model import GNN
from simg.graph_construction import convert_NBO_graph_to_downstream
import wandb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--train_ratio", type=float, help="Ratio of dataset for training")
    parser.add_argument("--val_ratio", type=float, help="Ratio of dataset for validation")
    parser.add_argument("--test_ratio", type=float, help="Ratio of dataset for testing")
    parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
    parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
    hparams = parser.parse_args()

    # load graph data
    logging.info(f'Loading graphs from {hparams.graphs_path}')
    data = torch.load(hparams.graphs_path)

    if hparams.from_NBO:
        logging.info('Converting to downstream format')
        data = [convert_NBO_graph_to_downstream(graph, molecular_only=hparams.molecular) for graph in tqdm(data)]

    # only get necessary attributes
    nice_graphs = []
    logging.info('Clean up')
    for graph in tqdm(data):
        nice_graphs.append(
            Data(
                x=torch.FloatTensor(graph.x),
                y=torch.FloatTensor(graph.y),
                edge_index=torch.LongTensor(graph.edge_index),
                edge_attr=torch.FloatTensor(graph.edge_attr),
                smiles=graph.smiles
            )
        )
    del data

    # set split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Compute the sizes for each split
    total_size = len(nice_graphs)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    print("Train:\t", train_size)
    print("Val:\t", val_size)
    print("Test:\t", test_size)

    # Perform the split
    train, val, test = random_split(nice_graphs, [train_size, val_size, test_size])

    # Save data
    os.makedirs("data/dataset", exist_ok=True)
    torch.save(train, "data/dataset/train.pt")
    torch.save(val, "data/dataset/val.pt")
    torch.save(test, "data/dataset/test.pt")