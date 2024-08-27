import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric import nn as geom_nn
from torch.optim import Adam
import os
from matplotlib import pyplot as plt
import wandb
import torch.distributed as dist

class GNNResModel(nn.Module):
    def __init__(self, input_features, edge_features, out_targets, hidden_size, fcn_hidden_dim, embedding_dim,
                 gnn_output_dim, heads):
        super().__init__()
        self.layers = []

        last_hs = input_features
        all_hss = 0

        for hs in hidden_size:
            self.layers += [
                geom_nn.GATConv(last_hs, hs, edge_dim=edge_features, heads=heads, concat=False),
                nn.ReLU(),
            ]

            last_hs = hs
            all_hss += hs

        self.layers.append(geom_nn.GCNConv(last_hs, gnn_output_dim))
        all_hss += gnn_output_dim

        self.layers = nn.ModuleList(self.layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(
                input_features + all_hss, fcn_hidden_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(fcn_hidden_dim),
            nn.Linear(fcn_hidden_dim, embedding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, out_targets),
        )

    def get_embedding(self, x, edge_index, edge_attr):
        each_step = [x]

        out = x

        for layer in self.layers:
            if isinstance(layer, geom_nn.MessagePassing):
                if isinstance(layer, geom_nn.GATConv):
                    out = layer(out, edge_index, edge_attr)
                else:
                    out = layer(out, edge_index)
            else:
                out = layer(out)
                each_step.append(out)

        each_step.append(out)

        out = torch.cat(each_step, dim=1)
        out = self.fcn_head(out)

        return out

    def forward(self, x, edge_index, edge_attr, batch):
        out = self.get_embedding(x, edge_index, edge_attr)

        out = geom_nn.global_mean_pool(out, batch)
        out = self.decoder(out)

        return out


class GNN(pl.LightningModule):
    def __init__(self, model_type, model_params, num_ffn_layers, target_dim, recalc_mae, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        model_dict = {
            'Res': GNNResModel,
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
                ffn_layers.append(nn.BatchNorm1d(num_features=model_params['hidden_channels']))
                ffn_layers.append(nn.ReLU())
            ffn_layers.append(nn.Linear(model_params['hidden_channels'], target_dim))
            self.ffn = nn.Sequential(*ffn_layers)

        self.loss = sid
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

        out = sigmoid(out)

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
            y, batch.y
        )

        if dist.get_rank() == 0:
            wandb.log({'loss': loss.item()})

        # wandb.log({'train_loss': loss})

        # for i in range(y.shape[0]):
        #     x_axis = torch.arange(400, 4000, step=4)
        #     epoch_dir = os.path.join("pics/train", f"epoch_{self.current_epoch}")
        #     os.makedirs(epoch_dir, exist_ok=True)
        #     plt.plot(x_axis, y[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
        #     plt.plot(x_axis, batch.y.reshape(y.shape)[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
        #     plt.legend()
        #     plt.savefig(os.path.join(epoch_dir, f"{batch_idx}_{i}.png"))
        #     plt.close()

        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch.y
        )

        # for i in range(y.shape[0]):
        #     x_axis = torch.arange(400, 4000, step=4)
        #     epoch_dir = os.path.join("pics/val", f"epoch_{self.current_epoch}")
        #     os.makedirs(epoch_dir, exist_ok=True)
        #     plt.plot(x_axis, y[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
        #     plt.plot(x_axis, batch.y.reshape(y.shape)[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
        #     plt.legend()
        #     plt.savefig(os.path.join(epoch_dir, f"{batch_idx}_{i}.png"))
        #     plt.close()

        wandb.log({'val_loss': loss})

        if self.best_val_loss is None or loss < self.best_val_loss:
            self.best_val_loss = loss
            wandb.log({'best_epoch': self.current_epoch})

        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y = self.forward(x, edge_index, edge_attr, batch.batch)

        loss = self.loss(
            y, batch.y
        )

        for i in range(y.shape[0]):
            x_axis = torch.arange(400, 4000, step=4)
            epoch_dir = os.path.join("pics/test", wandb.run.name)
            os.makedirs(epoch_dir, exist_ok=True)
            plt.plot(x_axis, y[i].detach().cpu().numpy(), color="#E8945A", label="Prediction")
            plt.plot(x_axis, batch.y.reshape(y.shape)[i].detach().cpu().numpy(), color="#5BB370", label="Ground Truth")
            plt.title(batch.smiles[i])
            plt.legend()
            plt.savefig(os.path.join(epoch_dir, f"{batch_idx}_{i}.png"))
            plt.close()

        wandb.log({'test_loss': loss})

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)

def sid(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-3, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    target_spectra = target_spectra.reshape(model_spectra.shape)

    # set limits
    model_spectra[model_spectra < threshold] = threshold
    target_spectra[target_spectra < threshold] = threshold

    # option 1: SID
    loss = torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)

    # option 2: MSE
    # loss = torch.square(target_spectra - model_spectra)
        
    loss = torch.sum(loss,axis=1)
    loss = loss.mean()

    return loss


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))