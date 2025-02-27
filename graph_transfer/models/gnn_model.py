import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, GPSConv
from torch_geometric.nn.resolver import activation_resolver
from torch.nn import Module, Linear, ModuleList
from typing import Optional


class BasicModel(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: Optional[int] = None,
                 num_layers: int = 1,
                 activation: str = 'tanh',
                 **kwargs) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.activation = activation_resolver(activation, **({}))

        self.inp = self.in_channels
        self.emb = None
        if self.hidden_channels is not None:
            self.emb = Linear(self.in_channels, self.hidden_channels)
            self.inp = self.hidden_channels

        self.conv = ModuleList()
        for _ in range(num_layers):
            self.conv.append(self.init_conv(self.inp, self.inp, activation, **kwargs))

        self.readout = Linear(self.inp, self.out_channels)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.emb(x) if self.emb else x
        for conv in self.conv:
            x = self.activation(conv(x, edge_index))
        x = self.readout(x)

        return x

    def init_conv(self, in_channels, out_channels, activation, **kwargs):
        raise NotImplementedError
    

class GCN_Model(BasicModel):
    def init_conv(self, in_channels, out_channels, activation, *args,  **kwargs):
        return GCNConv(in_channels, out_channels)


class GAT_Model(BasicModel):
    def init_conv(self, in_channels, out_channels, activation, *args,  **kwargs):
        return GATConv(in_channels, out_channels)
    

class GIN_Model(BasicModel):
    def init_conv(self, in_channels, out_channels, activation, *args, **kwargs):
        nn = Linear(in_channels, out_channels)
        return GINConv(nn, train_eps = True)

class SAGE_Model(BasicModel):
    def init_conv(self, in_channels, out_channels, activation, *args,  **kwargs):
        return SAGEConv(in_channels, out_channels)
    

class GPS_Model(BasicModel):
    def init_conv(self, in_channels, out_channels, activation, *args,  **kwargs):
        assert in_channels == out_channels
        attn_kwargs = {'dropout': 0.0}
        nn = GCNConv(in_channels, out_channels)
        return GPSConv(in_channels, nn, heads=2,
                        #attn_type='multihead', attn_kwargs=attn_kwargs, # pyg >= 2.4
                        attn_dropout=0.0, # pyg < 2.4.0
                        norm='layer')