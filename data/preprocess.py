import os
import sys
import torch
import argparse
from utils import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to save the PyG graphs")
    parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
    parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
    parser.add_argument("--size", type=int, help="Size of dataset")
    args = parser.parse_args()

    graphs = preprocess(args)
    torch.save(graphs, args.graphs_path)