import argparse
import yaml
import os
import random
import logging
import torch
import pytorch_lightning as pl
from tqdm.autonotebook import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import random_split
from model import GNN
from simg.graph_construction import convert_NBO_graph_to_downstream
import wandb
from torch.utils.data import Subset
import torch.distributed as dist

# if dist.get_rank() == 0:
#     wandb.init(project='simg-ir')

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
parser.add_argument("--split_dir", type=str, help="Path to the folder of the split dataset")
parser.add_argument("--model_config", type=str, help="Path to the config of the model")
parser.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")
parser.add_argument("--bs", type=int, help="Batch size")
hparams = parser.parse_args()

graphs = torch.load(os.path.join(hparams.graphs_path))

train_indices = torch.load(os.path.join(hparams.split_dir, "train_indices.pt"))
train_subset = Subset(graphs, train_indices)
train_loader = DataLoader(train_subset, batch_size=hparams.bs, shuffle=True, drop_last=True, num_workers=12)

val_indices = torch.load(os.path.join(hparams.split_dir, "val_indices.pt"))
val_subset = Subset(graphs, val_indices)
val_loader = DataLoader(val_subset, batch_size=hparams.bs, num_workers=12)

test_indices = torch.load(os.path.join(hparams.split_dir, "test_indices.pt"))
test_subset = Subset(graphs, test_indices)
test_loader = DataLoader(test_subset, batch_size=hparams.bs, num_workers=12)

# set up model config
with open(hparams.model_config, "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
if (not model_config['model_type'].endswith('_tg')) and model_config['model_type'] != 'FiLM':
    model_config['model_params']['input_features'] = graphs[0].x.shape[1]
    model_config['model_params']['out_targets'] = len(graphs[0].y.shape[1])
    model_config['model_params']['edge_features'] = graphs[0].edge_attr.shape[1]
else:
    model_config['model_params']['in_channels'] = graphs[0].x.shape[1]
    model_config['target_dim'] = graphs[0].y.shape[0]
    model_config['num_ffn_layers'] = 3

    if model_config['model_type'] == 'GAT_tg':
        model_config['model_params']['edge_dim'] = graphs[0].edge_attr.shape[1]
model_config['recalc_mae'] = None

gnn = GNN(**model_config)
gnn.train()
trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, strategy='ddp_find_unused_parameters_true')
trainer.fit(gnn, train_loader, val_loader)
trainer.test(gnn, test_loader)