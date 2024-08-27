import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from torchvision import datasets, transforms
import json

def train(rank, world_size, sweep_config):
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Initialize WandB only on the main process
    if rank == 0:
        wandb.init(config=sweep_config, project="your_project_name")
    
    # Set up model, dataset, and DataLoader
    model = MySmallModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=sampler, num_workers=4)
    
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    
    for epoch in range(wandb.config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch[0].to(rank), batch[1].to(rank)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        # Synchronize processes
        dist.barrier()
        
        # Log metrics only on the main process
        if rank == 0:
            avg_loss = epoch_loss / len(dataloader)
            wandb.log({"epoch": epoch, "train_loss": avg_loss})
    
    # Clean up
    dist.destroy_process_group()

def main():
    # run wandb sweep
    with open("configs/sweep_config.json", 'r') as file:
        sweep_config = json.load(file)

    sweep_id = wandb.sweep(sweep_config, project="simg_ir")
    wandb.agent(sweep_id, function=main, count=1000)
    
    def train_wrapper():
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size, sweep_config), nprocs=world_size, join=True)
    
    wandb.agent(sweep_id, function=train_wrapper)

if __name__ == "__main__":
    main()