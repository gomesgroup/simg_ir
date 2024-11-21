import argparse
import os
import logging
import torch
from tqdm.autonotebook import tqdm
from torch_geometric.data import Data
from torch.utils.data import random_split
from simg.graph_construction import convert_NBO_graph_to_downstream
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import gc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", type=str, help="Path to directory containing the PyG graphs")
    parser.add_argument("--train_ratio", type=float, help="Ratio of dataset for training")
    parser.add_argument("--val_ratio", type=float, help="Ratio of dataset for validation")
    parser.add_argument("--test_ratio", type=float, help="Ratio of dataset for testing")
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "scaffold"],
                      help="Split type: random or scaffold-based")
    hparams = parser.parse_args()

    nice_graphs = torch.load(os.path.join(hparams.graphs_dir, "graphs.pt"))
    total_size = len(nice_graphs)

    if hparams.split_type == "random":
        # Compute the sizes for each split
        train_size = int(hparams.train_ratio * total_size)
        val_size = int(hparams.val_ratio * total_size)
        test_size = total_size - train_size - val_size

        # only save indices for split
        train_indices, val_indices, test_indices = random_split(range(total_size), [train_size, val_size, test_size])

    else:  # scaffold split
        # Generate scaffolds for each molecule
        scaffolds = {}
        for i, graph in enumerate(nice_graphs):
            mol = Chem.MolFromSmiles(graph.smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [i]
            else:
                scaffolds[scaffold].append(i)

        # Sort scaffolds by size
        scaffold_sets = [scaffold_set for _, scaffold_set in sorted(scaffolds.items(), 
                        key=lambda x: len(x[1]), reverse=True)]
        
        train_cutoff = int(hparams.train_ratio * total_size)
        val_cutoff = int((hparams.train_ratio + hparams.val_ratio) * total_size)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) <= train_cutoff:
                train_indices.extend(scaffold_set)
            elif len(val_indices) + len(scaffold_set) <= val_cutoff - len(train_indices):
                val_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)

        # Convert to tensor subsets for consistency
        train_indices = torch.tensor(train_indices)
        val_indices = torch.tensor(val_indices)
        test_indices = torch.tensor(test_indices)

    # Save data
    os.makedirs(os.path.join(hparams.graphs_dir, "split"), exist_ok=True)
    torch.save(train_indices, os.path.join(hparams.graphs_dir, "split", "train_indices.pt"))
    torch.save(val_indices, os.path.join(hparams.graphs_dir, "split", "val_indices.pt"))
    torch.save(test_indices, os.path.join(hparams.graphs_dir, "split", "test_indices.pt"))