import argparse
import os
import logging
import torch
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data
from torch.utils.data import random_split
from simg.graph_construction import convert_NBO_graph_to_downstream
import gc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")

    parser.add_argument("--train_ratio", type=float, help="Ratio of dataset for training")
    parser.add_argument("--val_ratio", type=float, help="Ratio of dataset for validation")
    parser.add_argument("--test_ratio", type=float, help="Ratio of dataset for testing") 
    hparams = parser.parse_args()

    nice_graphs = torch.load(hparams.graphs_path)

    # Compute the sizes for each split
    total_size = len(nice_graphs)
    train_size = int(hparams.train_ratio * total_size)
    val_size = int(hparams.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # only save indices for split
    train_indices, val_indices, test_indices = random_split(range(total_size), [train_size, val_size, test_size])

    # Save data
    os.makedirs("data/split", exist_ok=True)
    torch.save(train_indices, "data/split/train_indices.pt")
    torch.save(val_indices, "data/split/val_indices.pt")
    torch.save(test_indices, "data/split/test_indices.pt")