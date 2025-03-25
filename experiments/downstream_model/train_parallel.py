import os
import yaml
import json
import torch
import argparse
import logging
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from model import GNN, corr_loss_fn
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(hparams, config):
    # Initialize WandB
    logging.info("WandB initialized")
    best_epoch = None
    best_val_loss = None
    
    # Load global best validation loss if it exists
    global_best_path = os.path.join(hparams.output_dir, "global_best_val_loss.pt")
    if os.path.exists(global_best_path):
        global_best_val_loss = torch.load(global_best_path).item()
        logging.info(f"Loaded global best validation loss: {global_best_val_loss}")
    else:
        global_best_val_loss = float('inf')
        logging.info("No previous global best validation loss found. Starting fresh.")
    
    # Set device
    device = torch.device(f'cuda:{hparams.gpu_ids}')
            
    # Set up model
    with open(hparams.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
        # Update model_params separately
        model_params = model_config['model_params']
        model_params['hidden_channels'] = config['hidden_channels']
        model_params['out_channels'] = config['hidden_channels']
        model_params['num_layers'] = config['num_layers']
        
        # Keep num_ffn_layers separate from model_params
        model_config['num_ffn_layers'] = config['num_ffn_layers']
        model_config['model_params'] = model_params
        model_config['recalc_mae'] = None
    
    gnn = GNN(**model_config).to(device)

    # Verify model is on the correct device
    model_device = next(gnn.parameters()).device
    print(f"Model is on device: {model_device}")
    assert str(model_device) == str(device), f"Model should be on {device} but is on {model_device}"

    optimizer = torch.optim.Adam(gnn.parameters(), lr=model_config['lr'])
    print(gnn)
            
    # Set up data
    graphs = torch.load(os.path.join(hparams.graphs_path), weights_only=False)

    train_indices = torch.load(os.path.join(hparams.split_dir, "train_indices.pt"), weights_only=False)
    train_subset = Subset(graphs, train_indices)
    
    val_indices = torch.load(os.path.join(hparams.split_dir, "val_indices.pt"), weights_only=False)
    val_subset = Subset(graphs, val_indices)

    train_loader = DataLoader(train_subset, batch_size=hparams.bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=hparams.bs, num_workers=0)

    # training loop
    for epoch in range(hparams.max_epochs):
        print("Epoch", epoch)
        
        # Training
        gnn.train()
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_ptr = batch.batch.to(device)
            targets = batch.y.to(device)
            
            outputs = gnn(x, edge_index, edge_attr, batch_ptr)
            loss = gnn.loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

            # Plot training spectra
            # plot_spectra(outputs, batch, epoch, batch_idx, "train", "single", "spectra")

            print("Loss: ", loss.item())
            wandb.log({"train_loss": loss.item()})
        
        # Validation
        print("Evaluating...")
        val_loss = 0.0
        gnn.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)
                batch_ptr = batch.batch.to(device)
                targets = batch.y.to(device)
                
                outputs = gnn(x, edge_index, edge_attr, batch_ptr)
                
                loss = gnn.loss_fn(outputs, batch)
                # corr_loss = corr_loss_fn(outputs, batch)
                val_loss += loss.item()      

                # plot_spectra(outputs, batch, epoch, batch_idx, "val", "whole", "diff")
            
            avg_val_loss = val_loss / len(val_loader)
            if best_val_loss is None or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                
                # Check if this is better than the global best
                if avg_val_loss < global_best_val_loss:
                    global_best_val_loss = avg_val_loss
                    # Save the model checkpoint
                    os.makedirs(hparams.output_dir, exist_ok=True)
                    checkpoint_path = os.path.join(hparams.output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': gnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'model_config': model_config
                    }, checkpoint_path)
                    # Save the global best validation loss
                    torch.save(torch.tensor(global_best_val_loss), global_best_path)
                    logging.info(f"New global best validation loss: {global_best_val_loss}. Model saved to {checkpoint_path}")
                    wandb.log({"global_best_val_loss": global_best_val_loss})

            wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "best_epoch": best_epoch})

    # report sweep objective
    wandb.log({"best_val_loss": best_val_loss})

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--split_dir", type=str, help="Path to the folder of the split dataset")
    parser.add_argument("--sweep_config", type=str, help="Path to the config file for the sweep")
    parser.add_argument("--model_config", type=str, help="Path to the config of the model")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--gpu_ids", type=str, help="GPU IDs")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    hparams = parser.parse_args()
    
    # Read the sweep configuration
    with open(hparams.sweep_config, 'r') as file:
        sweep_config = json.load(file)

    sweep_id = wandb.sweep(sweep_config, project="simg-ir")
    
    def train_wrapper():
        # Initialize WandB and get config
        wandb.init()
        config = {
            'hidden_channels': wandb.config.hidden_channels,
            'num_layers': wandb.config.num_layers,
            'num_ffn_layers': wandb.config.num_ffn_layers,
        }
    
        train(hparams, config)
    
    wandb.agent(sweep_id, function=train_wrapper, count=1000)

def plot_spectra(outputs, batch, epoch, batch_idx, split, quantity, type):
    if quantity == "single":
        batch_range = range(0, 1)
    elif quantity == "whole":
        batch_range = range(0, len(batch))

    batch_size = outputs.shape[0]
    dim = outputs.shape[1]
    if type == "diff":
        preds = outputs
        targets = batch.y.reshape(batch_size, dim)
        plt.ylim(-1.5, 1.5)
    # elif type == "spectra":
    #     gas_spectra = batch.gas_spectra.reshape(batch_size, dim).to(outputs.device)
    #     preds = gas_spectra + outputs
    #     targets = batch.liquid_spectra.reshape(batch_size, dim).to(outputs.device)  
    #     plt.ylim(0, 1.5)
    elif type == "spectra":
        preds = outputs
        targets = batch.y.reshape(batch_size, dim)  
        plt.ylim(0, 1)
    elif type == "both":
        dim = int(outputs.shape[1] / 2)
        pred_gas = outputs[:, :dim]
        pred_liq = outputs[:, dim:]
        pred_diff = pred_liq - pred_gas

        true_gas = batch.gas_spectra.reshape(batch_size, dim).to(outputs.device)
        true_liq = batch.liquid_spectra.reshape(batch_size, dim).to(outputs.device)
        true_diff = true_liq - true_gas
        
        preds = pred_diff
        targets = true_diff
        plt.ylim(-1.5, 1.5)
    
    dir = os.path.join("pics", wandb.run.name, split, f"epoch_{epoch}")
    os.makedirs(dir, exist_ok=True)

    for i in batch_range:
        x_axis = torch.arange(400, 4000, step=4)
        
        plt.plot(x_axis, preds[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
        plt.plot(x_axis, targets[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
        plt.legend()
        plt.savefig(os.path.join(dir, f"{batch_idx}_{i}.png"))
        plt.close()

if __name__ == "__main__":
    print("PID: ", os.getpid())
    main()