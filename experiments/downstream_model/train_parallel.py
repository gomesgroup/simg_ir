import os
import yaml
import json
import torch
import argparse
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric DataLoader
from model import GNN, sid
import wandb
import gc
from tqdm import tqdm

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size, hparams, config):
    try:
        # Setup DDP environment
        setup_ddp(rank, world_size)
        
        # Initialize WandB only on the main process
        if rank == 0:
            wandb.init(project='simg-ir', config=config)
            logging.info("WandB initialized")
        
        # Set up model
        with open(hparams.model_config, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
            model_config['model_params']['out_channels'] = config['out_channels']
            model_config['model_params']['num_layers'] = config['num_layers']
            model_config['recalc_mae'] = None
        
        gnn = GNN(**model_config, num_ffn_layers=config['num_ffn_layers']).to(rank)
        gnn = DDP(gnn, device_ids=[rank])
        
        # Set up data
        train_data = torch.load(os.path.join(hparams.dataset_dir, "train.pt"))
        val_data = torch.load(os.path.join(hparams.dataset_dir, "val.pt"))
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_data, batch_size=hparams.bs, shuffle=False, sampler=train_sampler, drop_last=False, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=hparams.bs, num_workers=0)
        
        optimizer = torch.optim.Adam(gnn.parameters(), lr=model_config['lr'])

        for epoch in range(hparams.max_epochs):
            print("Epoch", epoch)
            gnn.train()
            epoch_loss = 0.0
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                x, edge_index, edge_attr, batch_ptr = batch.x.to(rank), batch.edge_index.to(rank), batch.edge_attr.to(rank), batch.batch.to(rank)
                targets = batch.y.to(rank)
                outputs = gnn(x, edge_index, edge_attr, batch_ptr)
                loss = sid(model_spectra=outputs, target_spectra=targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                del x, edge_index, edge_attr, batch_ptr, targets, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()
            
            # Synchronize processes
            dist.barrier()

            # Validation step on rank 0
            if rank == 0:
                print("Evaluating...")
                gnn.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        x, edge_index, edge_attr, batch_ptr = batch.x.to(rank), batch.edge_index.to(rank), batch.edge_attr.to(rank), batch.batch.to(rank)
                        targets = batch.y.to(rank)
                        outputs = gnn(x, edge_index, edge_attr, batch_ptr)
                        loss = sid(model_spectra=outputs, target_spectra=targets)
                        val_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
                logging.info(f"Epoch {epoch}: Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # Clean up the process group
        dist.destroy_process_group()
    except:
        dist.destroy_process_group()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to the folder of the split dataset")
    parser.add_argument("--model_config", type=str, help="Path to the config of the model")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    hparams = parser.parse_args()
    
    # Read the sweep configuration
    with open("configs/sweep_config.json", 'r') as file:
        sweep_config = json.load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="simg_ir")
    
    # Start the sweep
    world_size = torch.cuda.device_count()
    
    def train_wrapper():
        # Initialize WandB and get config
        wandb.init()
        config = {
            'out_channels': wandb.config.out_channels,
            'num_layers': wandb.config.num_layers,
            'num_ffn_layers': wandb.config.num_ffn_layers,
        }
        mp.spawn(train, args=(world_size, hparams, config), nprocs=world_size, join=True)
    
    wandb.agent(sweep_id, function=train_wrapper, count=1000)

if __name__ == "__main__":
    main()
