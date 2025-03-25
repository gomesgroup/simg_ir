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
import math

class GNN(pl.LightningModule):
    # def __init__(self, model_type, model_params, num_ffn_layers, target_dim, recalc_mae, lr=2e-4, 
    #              curriculum_epochs=30, max_epochs=100, curriculum_strategy='threshold'):
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

        # # Curriculum learning parameters
        # self.curr_epoch = 0
        # self.curriculum_epochs = curriculum_epochs
        # self.max_epochs = max_epochs
        # self.curriculum_strategy = curriculum_strategy
        
        # # Initial threshold for curriculum learning
        # self.initial_threshold = 0.5  # Start with only the most prominent peaks
        # self.final_threshold = 0.05   # End with including most peaks
        
        # # Initial gamma for focal loss
        # self.initial_gamma = 2.0
        # self.final_gamma = 4.0
        
        self.loss_fn = sid_loss_fn
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

        # out = torch.tanh(out)
        out = torch.sigmoid(out)

        return out
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
        
    # def get_curriculum_params(self):
    #     """Calculate curriculum parameters based on current epoch"""
    #     # Calculate progress through curriculum (0 to 1)
    #     progress = min(self.curr_epoch / self.curriculum_epochs, 1.0)
        
    #     # Smoothly transition from initial to final values
    #     threshold = self.initial_threshold - progress * (self.initial_threshold - self.final_threshold)
    #     gamma = self.initial_gamma + progress * (self.final_gamma - self.initial_gamma)
        
    #     # Weight for combining losses (gradually increase importance of correlation)
    #     corr_weight = progress * 0.3  # Max weight of 0.3 for correlation loss
        
    #     return threshold, gamma, corr_weight
        
    # def curriculum_loss(self, preds, batch):
    #     """Curriculum learning loss function that adapts based on training progress"""
    #     threshold, gamma, corr_weight = self.get_curriculum_params()
        
    #     # Log current curriculum parameters
    #     self.log('curriculum_threshold', threshold)
    #     self.log('curriculum_gamma', gamma)
    #     self.log('corr_weight', corr_weight)
        
    #     # Calculate focal loss with current threshold and gamma
    #     focal = self.adaptive_focal_loss(preds, batch, gamma, threshold)
        
    #     # If we're in later stages, also include correlation loss
    #     if corr_weight > 0:
    #         corr = corr_loss_fn(preds, batch)
    #         loss = (1 - corr_weight) * focal + corr_weight * corr
            
    #         # Log individual loss components
    #         self.log('focal_loss', focal)
    #         self.log('corr_loss', corr)
    #     else:
    #         loss = focal
            
    #     return loss
        
    # def adaptive_focal_loss(self, preds, batch, gamma, threshold):
    #     """Focal loss with adaptive threshold based on curriculum progress"""
    #     batch_size = preds.shape[0]
    #     dim = preds.shape[1]
    #     target = batch.y.reshape(batch_size, dim).to(preds.device)

    #     error = torch.abs(preds - target)
        
    #     # Focus on peaks above the current threshold
    #     peak_mask = torch.abs(target) > threshold
        
    #     # Apply different scaling based on peak importance
    #     scale_factor = peak_mask.float() * (error ** gamma) + (~peak_mask).float() * 0.1
        
    #     # Calculate weighted loss
    #     loss = (scale_factor * error**2).mean()
        
    #     # Log percentage of peaks being considered
    #     peak_percentage = peak_mask.float().mean() * 100
    #     self.log('peak_percentage', peak_percentage)
        
    #     return loss
        
    # def log_peak_metrics(self, preds, batch):
    #     """Log metrics specific to different peak intensity ranges"""
    #     batch_size = preds.shape[0]
    #     dim = preds.shape[1]
    #     target = batch.y.reshape(batch_size, dim).to(preds.device)
        
    #     # Define peak intensity ranges
    #     ranges = [(0.5, 1.0), (0.2, 0.5), (0.1, 0.2), (0.05, 0.1), (0, 0.05)]
        
    #     for min_val, max_val in ranges:
    #         # Create mask for peaks in this range
    #         mask = (torch.abs(target) >= min_val) & (torch.abs(target) < max_val)
            
    #         if mask.sum() > 0:
    #             # Calculate MAE for peaks in this range
    #             mae = torch.abs(preds[mask] - target[mask]).mean()
    #             self.log(f'mae_peaks_{min_val:.2f}_{max_val:.2f}', mae)
    
    # def increment_epoch(self):
    #     self.curr_epoch += 1

def mse_loss_fn(preds, batch):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    loss = torch.nn.functional.mse_loss(preds, target)

    return loss

def sid_loss_fn(preds, batch, threshold=1e-3):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    preds = torch.where(preds < threshold, threshold, preds)
    target = torch.where(target < threshold, threshold, target)

    loss = torch.mul(torch.log(torch.div(preds,target)),preds) + torch.mul(torch.log(torch.div(target,preds)),target)
    loss = torch.sum(loss,axis=1)

    loss = loss.mean()

    return loss

def sid_loss_both_fn(preds, batch, threshold=1e-3):
    batch_size = preds.shape[0]
    # dim = preds.shape[1]
    dim = int(preds.shape[1] / 2)
    # Split predictions into gas and liquid components
    pred_gas = preds[:, :dim]  # First 900 dimensions for gas
    pred_liq = preds[:, dim:]  # Remaining 900 dimensions for liquid
    
    # pred_liquid_spectra = preds + batch.gas_spectra.reshape(batch_size, dim).to(preds.device)
    # true_liquid_spectra = batch.liquid_spectra.reshape(batch_size, dim).to(preds.device)
    
    # pred_liquid_spectra = preds
    # true_liquid_spectra = batch.y.reshape(batch_size, dim).to(preds.device)

    true_gas = batch.gas_spectra.reshape(batch_size, dim).to(preds.device)
    true_liq = batch.liquid_spectra.reshape(batch_size, dim).to(preds.device)

    pred_gas = torch.where(pred_gas < threshold, threshold, pred_gas)
    pred_liq = torch.where(pred_liq < threshold, threshold, pred_liq)
    true_gas = torch.where(true_gas < threshold, threshold, true_gas)
    true_liq = torch.where(true_liq < threshold, threshold, true_liq)

    gas_loss = torch.mul(torch.log(torch.div(pred_gas,true_gas)),pred_gas) + torch.mul(torch.log(torch.div(true_gas,pred_gas)),true_gas)
    liq_loss = torch.mul(torch.log(torch.div(pred_liq,true_liq)),pred_liq) + torch.mul(torch.log(torch.div(true_liq,pred_liq)),true_liq)
    gas_loss = gas_loss.sum(axis=1)
    liq_loss = liq_loss.sum(axis=1)

    loss = gas_loss + liq_loss
    loss = loss.mean()

    return loss

def sid_exp_loss_fn(preds, batch, threshold=1e-3):
    batch_size = preds.shape[0]
    dim = preds.shape[1]

    preds = torch.exp(preds)
    target = torch.exp(batch.y.reshape(batch_size, dim).to(preds.device))

    preds = torch.where(preds < threshold, threshold, preds)
    target = torch.where(target < threshold, threshold, target)

    loss = torch.mul(torch.log(torch.div(preds,target)),preds) + torch.mul(torch.log(torch.div(target,preds)),target)
    
    loss = torch.sum(loss,axis=1)
    loss = loss.mean()

    return loss

def focal_loss_fn(preds, batch, gamma=3.0, threshold=0.1):
    batch_size = preds.shape[0]
    dim = preds.shape[1]
    target = batch.y.reshape(batch_size, dim).to(preds.device)

    error = torch.abs(preds - target)
    scale_factor = ((torch.abs(target) > threshold).float() * (error ** gamma) 
                    + (torch.abs(target) <= threshold).float())
    loss = (scale_factor * error**2).mean()

    return loss

def corr_loss_fn(preds, batch):
    batch_size = preds.shape[0]
    dim = preds.shape[1]

    gas_spectra = batch.gas_spectra.reshape(batch_size, dim).to(preds.device)
    true_liquid_spectra = batch.liquid_spectra.reshape(batch_size, dim).to(preds.device)
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