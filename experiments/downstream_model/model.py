import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric import nn as geom_nn
from torch.optim import Adam
import os
from matplotlib import pyplot as plt
import wandb
import torch.distributed as dist
import json
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from rdkit import Chem
from rdkit.Chem import AllChem

class GNN(pl.LightningModule):
    def __init__(self, model_type, model_params, num_ffn_layers, target_dim, recalc_mae, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        model_dict = {
            'PNA_tg': geom_nn.PNA,
            'GAT_tg': geom_nn.GAT,
            'GCN_tg': geom_nn.GraphSAGE,
        }

        if model_type in model_dict:
            self.model = model_dict[model_type](**model_params, norm='batch')
        else:
            raise ValueError(f'Model type {model_type} not supported')

        self.model_type = model_type
        if model_type in ["GCN_tg", "GAT_tg", 'FiLM']:
            # create feed-forward network
            ffn_layers = []
            for _ in range(num_ffn_layers):
                ffn_layers.append(nn.Linear(model_params['hidden_channels'], model_params['hidden_channels']))
                ffn_layers.append(nn.BatchNorm1d(model_params['hidden_channels']))
                ffn_layers.append(nn.ReLU())
            ffn_layers.append(nn.Linear(model_params['hidden_channels'], target_dim))
            self.ffn = nn.Sequential(*ffn_layers)

        self.loss_fn = mse_loss
        self.recal_mae = recalc_mae
        self.best_val_loss = None

    def get_embedding(self, x, edge_index, edge_attr):
        if self.model_type in ['GCN_tg', 'FiLM']:
            return self.model(x, edge_index)
        if self.model_type == 'GAT_tg':
            return self.model(x, edge_index, edge_attr)

        return self.model.get_embedding(x, edge_index, edge_attr)

    def forward(self, x, edge_index, edge_attr, batch):
        if self.model_type in ['GCN_tg', 'FiLM']:
            out = self.model(x, edge_index)
            out = geom_nn.global_mean_pool(out, batch)
            out = self.ffn(out)
        elif self.model_type == 'GAT_tg':
            out = self.model(x, edge_index, edge_attr)
            out = geom_nn.global_mean_pool(out, batch)
            out = self.ffn(out)
        else:
            out = self.model(x, edge_index, edge_attr, batch)

        out = torch.tanh(out)

        return out

    def log_scalar_dict(self, metrics, prefix):
        for k, v in metrics.items():
            self.log(
                f'{prefix}_{k}',
                v
            )

    def log_maes(self, y_true, y_pred, predix):
        for i in range(y_true.shape[1]):
            self.log(
                f'{predix}_mae_{i}',
                torch.mean(torch.abs(y_true[:, i] - y_pred[:, i])).item() / self.recal_mae[i]
            )

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch
        )

        print("Train loss: ", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch
        )

        # wandb.log({'val_loss': loss})

        # if self.best_val_loss is None or loss < self.best_val_loss:
        #     self.best_val_loss = loss
        #     wandb.log({'best_epoch': self.current_epoch})

        print("Val loss: ", loss.item())

        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch
        )

        # wandb.log({'test_loss': loss})

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

def mse_loss(preds, batch):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    loss = torch.nn.functional.mse_loss(preds, target)

    return loss

def focal_loss(preds, batch, gamma=2.0, threshold=0.1):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    error = torch.abs(preds - target)
    scale_factor = (torch.abs(target) > threshold).float() * (error ** gamma) + (torch.abs(target) <= threshold).float()
    loss = (scale_factor * error**2).mean()

    return loss

def corr_loss(preds, batch):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    gas_spectra = batch.gas_spectra.reshape(batch_size, dim)
    true_liquid_spectra = batch.liquid_spectra.reshape(batch_size, dim)
    pred_liquid_spectra = gas_spectra + preds

    corrs = torch.zeros(batch_size, device=preds.device)
    for i in range(batch_size):
        # Stack the two vectors to create a 2×dim tensor
        stacked = torch.stack([pred_liquid_spectra[i], true_liquid_spectra[i]])
        # Compute correlation matrix (2×2)
        corr_matrix = torch.corrcoef(stacked)
        # Extract the correlation coefficient (off-diagonal element)
        corrs[i] = corr_matrix[0, 1]

    # We want to maximize correlation, so we return 1 - corr
    return (1 - corrs).mean()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))