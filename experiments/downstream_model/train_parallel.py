import os
import yaml
import json
import torch
import argparse
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric DataLoader
from model import GNN, sid
import wandb
import gc
from tqdm import tqdm
from matplotlib import pyplot as plt

# mp.set_sharing_strategy('file_system')

def setup_ddp(rank, world_size, gpu_ids):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_ids[rank])

def train(rank, world_size, hparams, config, gpu_ids):
    try:
        # Setup DDP environment
        setup_ddp(rank, world_size, gpu_ids)
        
        # Initialize WandB only on the main process
        if rank == 0:
            wandb.init(project='simg-ir', config=config)
            logging.info("WandB initialized")
            best_epoch = None
            best_val_loss = None
            
        # Set up model
        with open(hparams.model_config, "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
            model_config['model_params']['hidden_channels'] = config['hidden_channels']
            model_config['model_params']['out_channels'] = config['hidden_channels']
            model_config['model_params']['num_layers'] = config['num_layers']
            model_config['recalc_mae'] = None
        gnn = GNN(**model_config, num_ffn_layers=config['num_ffn_layers']).to(gpu_ids[rank])
        gnn = DDP(gnn, device_ids=[gpu_ids[rank]])
        optimizer = torch.optim.Adam(gnn.parameters(), lr=model_config['lr'])
        print(gnn)
                
        # Set up data
        graphs = torch.load(os.path.join(hparams.graphs_path))

        train_indices = torch.load(os.path.join(hparams.split_dir, "train_indices.pt"))
        train_subset = Subset(graphs, train_indices)
        
        val_indices = torch.load(os.path.join(hparams.split_dir, "val_indices.pt"))
        val_subset = Subset(graphs, val_indices)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_subset, batch_size=hparams.bs, shuffle=False, sampler=train_sampler, drop_last=False, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=hparams.bs, num_workers=0)

        # training loop
        for epoch in range(hparams.max_epochs):
            print("Epoch", epoch)
            train_sampler.set_epoch(epoch)
            
            gnn.train()
            for batch in tqdm(train_loader):
                optimizer.zero_grad()

                x, edge_index, edge_attr, batch_ptr = batch.x.to(gpu_ids[rank]), batch.edge_index.to(gpu_ids[rank]), batch.edge_attr.to(gpu_ids[rank]), batch.batch.to(gpu_ids[rank])
                targets = batch.y.to(gpu_ids[rank])

                outputs = gnn(x, edge_index, edge_attr, batch_ptr)
                loss = sid(model_spectra=outputs, target_spectra=targets)
                loss.backward()
                optimizer.step()

                # for i in range(10):
                #     x_axis = torch.arange(400, 4000, step=4)
                #     epoch_dir = os.path.join("pics/train", f"epoch_{epoch}")
                #     os.makedirs(epoch_dir, exist_ok=True)
                #     plt.plot(x_axis, outputs[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
                #     plt.plot(x_axis, batch.y.reshape(outputs.shape)[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
                #     plt.legend()
                #     plt.savefig(os.path.join(epoch_dir, f"{i}.png"))
                #     plt.close()

                if rank == 0:
                    wandb.log({"train_loss": loss})
            
            # Synchronize processes
            dist.barrier()

            # Validation step w/o parallelization
            if rank == 0:
                print("Evaluating...")
                val_loss = 0.0

                # gnn.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x, edge_index, edge_attr, batch_ptr = batch.x.to(gpu_ids[rank]), batch.edge_index.to(gpu_ids[rank]), batch.edge_attr.to(gpu_ids[rank]), batch.batch.to(gpu_ids[rank])
                        targets = batch.y.to(gpu_ids[rank])
                        outputs = gnn(x, edge_index, edge_attr, batch_ptr)
                        loss = sid(model_spectra=outputs, target_spectra=targets)
                        val_loss += loss.item()      

                        # for i in range(outputs.shape[0]):
                        #     x_axis = torch.arange(400, 4000, step=4)
                        #     epoch_dir = os.path.join("pics/val", f"epoch_{epoch}")
                        #     os.makedirs(epoch_dir, exist_ok=True)
                        #     plt.plot(x_axis, outputs[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
                        #     plt.plot(x_axis, batch.y.reshape(outputs.shape)[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
                        #     plt.legend()
                        #     plt.savefig(os.path.join(epoch_dir, f"{i}.png"))
                        #     plt.close()
                
                avg_val_loss = val_loss / len(val_loader)
                if best_val_loss is None or avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch

                wandb.log({"epoch": epoch, "val_loss": avg_val_loss, "best_epoch": best_epoch})

        # report sweep objective
        if rank == 0:
            wandb.log({"best_val_loss": best_val_loss})

        # Clean up the process group
        dist.destroy_process_group()
    except Exception as e:
        print(e)
        dist.destroy_process_group()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_path", type=str, help="Path to the PyG graphs")
    parser.add_argument("--split_dir", type=str, help="Path to the folder of the split dataset")
    parser.add_argument("--model_config", type=str, help="Path to the config of the model")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of training epochs")
    parser.add_argument("--bs", type=int, help="Batch size")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use")
    hparams = parser.parse_args()
    
    # Read the sweep configuration
    with open("configs/sweep_config.json", 'r') as file:
        sweep_config = json.load(file)
    
    sweep_id = wandb.sweep(sweep_config, project="simg-ir")
    
    # Parse GPU IDs
    gpu_ids = [int(id) for id in hparams.gpu_ids.split(",")]
    world_size = len(gpu_ids)

    def train_wrapper():
        # Initialize WandB and get config
        wandb.init()
        config = {
            'hidden_channels': wandb.config.hidden_channels,
            'num_layers': wandb.config.num_layers,
            'num_ffn_layers': wandb.config.num_ffn_layers,
        }

        mp.spawn(train, args=(world_size, hparams, config, gpu_ids), nprocs=world_size, join=True)
    
    wandb.agent(sweep_id, function=train_wrapper, count=1000)

if __name__ == "__main__":
    print("PID: ", os.getpid())
    main()