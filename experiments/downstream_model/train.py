# import argparse
# import yaml
# import os
# import random
# import logging
# import torch
# import pytorch_lightning as pl
# from tqdm.autonotebook import tqdm
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# from torch.utils.data import random_split
# from model import GNN
# from simg.graph_construction import convert_NBO_graph_to_downstream
# import wandb

# wandb.init(project='simg-ir')

# logging.basicConfig(level=logging.INFO)

# parser = argparse.ArgumentParser()
# parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
# parser.add_argument( "--splits_path", type=str, help="Path to the splits. Must contain {train,val,test}.txt files.")
# parser.add_argument("--bs", type=int, help="Batch size")
# parser.add_argument("--model_config", type=str, help="Path to the config of the model")
# parser.add_argument("--sample", type=float, default=1.0, help='Do sampling?')
# parser.add_argument("--parts", action="store_true", help='In parts?')
# parser.add_argument("--add_s", type=str, default="", help='Suffix to add to the name')
# parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
# parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
# hparams = parser.parse_args()

# # Check inputs
# if hparams.parts:
#     assert os.path.exists(hparams.graphs_path), "Graphs path does not exist"
# assert os.path.exists(hparams.model_config), "Model config does not exist"

# logging.info(f'Loading graphs from {hparams.graphs_path}')
# if not hparams.parts:
#     data = torch.load(hparams.graphs_path)
# else:
#     data = sum([
#         torch.load(os.path.join(hparams.graphs_path, file)) for file in tqdm(os.listdir(hparams.graphs_path)) if
#         file.endswith('.pt')
#     ], [])

# if hparams.from_NBO:
#     logging.info('Converting to downstream format')
#     data = [convert_NBO_graph_to_downstream(graph, molecular_only=hparams.molecular) for graph in tqdm(data)]

# nice_graphs = []
# logging.info('Clean up')
# for graph in tqdm(data):
#     nice_graphs.append(
#         Data(
#             x=torch.FloatTensor(graph.x),
#             y=torch.FloatTensor(graph.y),
#             edge_index=torch.LongTensor(graph.edge_index),
#             edge_attr=torch.FloatTensor(graph.edge_attr)
#         )
#     )
# del data

# # Set your split ratios
# train_ratio = 0.8
# val_ratio = 0.1
# test_ratio = 0.1

# # Compute the sizes for each split
# total_size = len(nice_graphs)
# train_size = int(train_ratio * total_size)
# val_size = int(val_ratio * total_size)
# test_size = total_size - train_size - val_size
# print("Train:\t", train_size)
# print("Val:\t", val_size)
# print("Test:\t", test_size)

# # Perform the split
# train, val, test = random_split(nice_graphs, [train_size, val_size, test_size])
# train_loader = DataLoader(train, batch_size=hparams.bs, shuffle=True, drop_last=False, num_workers=12)
# val_loader = DataLoader(val, batch_size=hparams.bs, num_workers=12)
# test_loader = DataLoader(test, batch_size=hparams.bs, num_workers=12)

# # set up model config
# with open(hparams.model_config, "r") as f:
#     model_config = yaml.load(f, Loader=yaml.FullLoader)
# if (not model_config['model_type'].endswith('_tg')) and model_config['model_type'] != 'FiLM':
#     model_config['model_params']['input_features'] = nice_graphs[0].x.shape[1]
#     model_config['model_params']['out_targets'] = len(nice_graphs[0].y.shape[1])
#     model_config['model_params']['edge_features'] = nice_graphs[0].edge_attr.shape[1]
# else:
#     model_config['model_params']['in_channels'] = nice_graphs[0].x.shape[1]
#     model_config['target_dim'] = nice_graphs[0].y.shape[0]
#     model_config['num_ffn_layers'] = 3

#     if model_config['model_type'] == 'GAT_tg':
#         model_config['model_params']['edge_dim'] = nice_graphs[0].edge_attr.shape[1]
# model_config['recalc_mae'] = None

# gnn = GNN(**model_config)
# gnn.train()
# trainer = pl.Trainer(accelerator='gpu', devices=4, enable_checkpointing=False)
# trainer.fit(gnn, train_loader, val_loader)
# trainer.test(gnn, test_loader)

# ------------------------------------- Hyperparameter optimization ----------------

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
from model import GNN
from simg.graph_construction import convert_NBO_graph_to_downstream
import wandb

def main():
    wandb.init(project='simg-ir', config=wandb.config)

    # set up model config
    with open(hparams.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    if (not model_config['model_type'].endswith('_tg')) and model_config['model_type'] != 'FiLM':
        model_config['model_params']['input_features'] = nice_graphs[0].x.shape[1]
        model_config['model_params']['out_targets'] = len(nice_graphs[0].y.shape[1])
        model_config['model_params']['edge_features'] = nice_graphs[0].edge_attr.shape[1]
    else:
        model_config['model_params']['in_channels'] = nice_graphs[0].x.shape[1]
        model_config['target_dim'] = nice_graphs[0].y.shape[0]

        model_config['model_params']['out_channels'] = wandb.config.out_channels
        model_config['model_params']['num_layers'] = wandb.config.num_layers

        if model_config['model_type'] == 'GAT_tg':
            model_config['model_params']['edge_dim'] = nice_graphs[0].edge_attr.shape[1]
    model_config['recalc_mae'] = None

    # train and evaluate
    gnn = GNN(**model_config, num_ffn_layers=wandb.config.num_ffn_layers)
    gnn.train()
    trainer = pl.Trainer(accelerator='gpu', devices=4, strategy="ddp", enable_checkpointing=False, max_epochs=hparams.max_epochs)
    trainer.fit(gnn, train_loader, val_loader)
    trainer.test(gnn, test_loader)

    wandb.log({'best_val_loss_model': gnn.best_val_loss})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--model_config", type=str, help="Path to the config of the model")
    parser.add_argument("--parts", action="store_true", help='In parts?')
    parser.add_argument("--from_NBO", action="store_true", help='Use NBO targets')
    parser.add_argument("--molecular", action="store_true", help='Transform graphs to molecular')
    hparams = parser.parse_args()

    # Check inputs
    if hparams.parts:
        assert os.path.exists(hparams.graphs_path), "Graphs path does not exist"
    assert os.path.exists(hparams.model_config), "Model config does not exist"

    # load graph data
    logging.info(f'Loading graphs from {hparams.graphs_path}')
    if not hparams.parts:
        data = torch.load(hparams.graphs_path)
    else:
        data = sum([
            torch.load(os.path.join(hparams.graphs_path, file)) for file in tqdm(os.listdir(hparams.graphs_path)) if
            file.endswith('.pt')
        ], [])

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
    train_loader = DataLoader(train, batch_size=hparams.bs, shuffle=True, drop_last=False, num_workers=12)
    val_loader = DataLoader(val, batch_size=hparams.bs, num_workers=12)
    test_loader = DataLoader(test, batch_size=hparams.bs, num_workers=12)

    # run wandb sweep
    with open("configs/sweep_config.json", 'r') as file:
        sweep_config = json.load(file)

    sweep_id = wandb.sweep(sweep_config, project="simg_ir")
    wandb.agent(sweep_id, function=main, count=1000)