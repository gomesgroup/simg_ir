import os
import sys
import torch
import argparse
from utils import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Type of dataset to preprocess")
    parser.add_argument("--graphs_path", type=str, help="Path to save the PyG graphs")
    parser.add_argument("--filename", type=str, help="Path to input data file")
    parser.add_argument("--gas_filename", type=str, help="Path to gas data file")
    parser.add_argument("--liquid_filename", type=str, help="Path to liquid data file")
    parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
    parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
    parser.add_argument("--phase", type=str, help="Phase of the simulated data", default=None)
    args = parser.parse_args()

    graphs = preprocess(args)
    os.makedirs(os.path.dirname(args.graphs_path), exist_ok=True)
    torch.save(graphs, args.graphs_path)