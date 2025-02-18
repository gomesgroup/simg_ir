import os
import yaml
import torch
import random
import pandas as pd
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from experiments.downstream_model.model import GNN, sid
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

import os
print(f"PID: {os.getpid()}")


def setup_tracking_molecules(graphs_path, n_molecules=5):
    """Randomly select n molecules from the dataset and create tracking folders"""
    # Load SMILES data
    with open(graphs_path, 'rb') as f:
        data = torch.load(f)
    selected_smiles = random.sample([graph.smiles for graph in data], n_molecules)
    
    # # Create folders for each SMILES
    # base_dir = "molecule_tracking"
    # os.makedirs(base_dir, exist_ok=True)
    
    # for smiles in selected_smiles:
    #     smiles_dir = os.path.join(base_dir, smiles.replace('/', '_'))
    #     os.makedirs(smiles_dir, exist_ok=True)
    
    return selected_smiles

def plot_spectrum(outputs, targets, epoch, smiles, x_axis, loss):
    """Plot and save prediction vs ground truth spectrum"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, outputs.detach().cpu().numpy(), color="#E8945A", label="Prediction")
    plt.plot(x_axis, targets.detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
    plt.legend()
    plt.title(f"Epoch {epoch} - Loss: {loss:.6f}")
    
    # Create a folder named with the loss value
    loss_folder = f"{loss:.6f}"
    save_dir = os.path.join("molecule_tracking", smiles.replace('/', '_'), loss_folder)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"spectrum.png")
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    config = {
        'hidden_channels': 1024,
        'num_layers': 8,
        'num_ffn_layers': 8
    }
    
    # Paths and parameters
    graphs_path = "data/datasets/savoie_649/graphs.pt"  # Update with your actual path
    split_dir = "data/datasets/savoie_649/split"       # Update with your actual path
    model_config_path = "configs/model_config_GCN_tg.yaml"  # Update with your actual path
    
    # Set device to GPU 0
    device = torch.device('cuda:1')
    
    # Select tracking molecules
    tracking_smiles = setup_tracking_molecules(graphs_path)
    
    # Load model configuration
    with open(model_config_path, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config['model_params']['hidden_channels'] = config['hidden_channels']
        model_config['model_params']['out_channels'] = config['hidden_channels']
        model_config['model_params']['num_layers'] = config['num_layers']
        model_config['recalc_mae'] = None
    
    # Initialize model
    model = GNN(**model_config, num_ffn_layers=config['num_ffn_layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    
    # Load data
    graphs = torch.load(graphs_path)
    train_indices = torch.load(os.path.join(split_dir, "train_indices.pt"))
    val_indices = torch.load(os.path.join(split_dir, "val_indices.pt"))
    
    train_subset = Subset(graphs, train_indices)
    val_subset = Subset(graphs, val_indices)
    
    batch_size = 4  # Set consistent batch size
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Training loop
    max_epochs = 100  # Adjust as needed
    x_axis = torch.arange(400, 4000, step=4)
    train_losses = []
    val_losses = []
    best_losses = {}  # Dictionary to track best loss for each SMILES
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch}")
        
        # Training
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.reshape(outputs.shape)
            train_loss = sid(model_spectra=outputs, target_spectra=targets)
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

            # Calculate individual losses and plot spectra for each molecule
            for i in range(outputs.shape[0]):
                individual_loss = sid(
                    model_spectra=outputs[i].unsqueeze(0), 
                    target_spectra=targets[i].unsqueeze(0)
                ).item()

                # Update best loss for this SMILES
                if batch[i].smiles not in best_losses or individual_loss < best_losses[batch[i].smiles]:
                    best_losses[batch[i].smiles] = individual_loss

                if batch[i].smiles in tracking_smiles:
                    # Create directory for plots
                    plot_dir = os.path.join("track_loss", batch[i].smiles)
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    # Plot spectra
                    plt.plot(x_axis, outputs[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
                    plt.plot(x_axis, targets[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
                    plt.legend()
                    plt.title(f"Loss: {individual_loss:.6f}")
                    
                    # Save plot with loss value as filename
                    plt.savefig(os.path.join(plot_dir, f"{individual_loss:.6f}.png"))
                    plt.close()

        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                targets = batch.y.reshape(outputs.shape)
                val_loss = sid(model_spectra=outputs, target_spectra=targets)
                total_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
    
    # Save losses to CSV
    pd.DataFrame({'train_loss': train_losses, 'train_step': range(len(train_losses))}).to_csv('losses.csv', index=False)
    pd.DataFrame({'val_loss': val_losses, 'epoch': range(len(val_losses))}).to_csv('val_losses.csv', index=False)
    pd.DataFrame({'smiles': list(best_losses.keys()), 'best_loss': list(best_losses.values())}).to_csv('best_losses.csv', index=False)

if __name__ == "__main__":
    main()
