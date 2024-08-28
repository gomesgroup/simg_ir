import argparse
import os
import logging
import torch
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data
from torch.utils.data import random_split
from simg.graph_construction import convert_NBO_graph_to_downstream
import gc

def load_data_in_batches(path, batch_size=10_000):
    data = torch.load(path)
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")

    parser.add_argument("--train_ratio", type=float, help="Ratio of dataset for training")
    parser.add_argument("--val_ratio", type=float, help="Ratio of dataset for validation")
    parser.add_argument("--test_ratio", type=float, help="Ratio of dataset for testing")
    # parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
    # parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
    hparams = parser.parse_args()

    # # load graph data
    # nice_graphs = []
    # for batch in tqdm(load_data_in_batches(hparams.graphs_path)):
    #     if hparams.from_NBO:
    #         logging.info('Converting to downstream format')
    #         batch = [convert_NBO_graph_to_downstream(graph, molecular_only=hparams.molecular) for graph in batch]
        
    #     logging.info('Clean up')
    #     for graph in batch:
    #         nice_graphs.append(
    #             Data(
    #                 x=torch.FloatTensor(graph.x),
    #                 y=graph.y,
    #                 edge_index=torch.LongTensor(graph.edge_index),
    #                 # edge_attr=torch.FloatTensor(graph.edge_attr),
    #                 smiles=graph.smiles
    #             )
    #         )

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