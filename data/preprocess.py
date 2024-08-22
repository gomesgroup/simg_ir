import torch
import argparse
from utils import preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, help="Size of dataset")
    args = parser.parse_args()

    graphs = preprocess(args.size)
    torch.save(graphs, "data/graphs.pt")
