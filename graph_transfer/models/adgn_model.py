import torch

from torch.nn import Parameter, init
from torch_geometric.nn import MessagePassing, GCNConv
from typing import Optional
import math
from models.ausiliar_modules import NaiveAggr
from models.gnn_model import BasicModel



conv_names = ['NaiveAggr', 'GCNConv']
class AntiSymmetricConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 graph_conv: str = 'NaiveAggr',
                 bias: bool = True) -> None:

        super().__init__(aggr = 'add')
        self.W = Parameter(torch.empty((in_channels, in_channels)))
        self.bias = Parameter(torch.empty(in_channels)) if bias else None

        if graph_conv == 'NaiveAggr':
            self.conv = NaiveAggr(in_channels)
        elif graph_conv == 'GCNConv':
            self.conv = GCNConv(in_channels, in_channels, bias=False)
        else:
            NotImplementedError(f'{graph_conv} not implemented. {graph_conv} is not in {conv_names}')

        self.graph_conv = graph_conv
        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)
        self.activ_fun= activ_fun

        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.antisymmetric_W = self.W - self.W.T - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        for i in range(self.num_iters):
            neigh_x = self.conv(x, edge_index=edge_index, edge_weight=edge_weight)
            conv = x @ self.antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)

        return x

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)



class ADGN_Model(BasicModel):
    def init_conv(self, in_channels: int, out_channels: int, activation:str, **kwargs) -> MessagePassing:
            return AntiSymmetricConv(in_channels=in_channels,
                            num_iters=kwargs['num_iters'],
                            gamma=kwargs['gamma'],
                            epsilon=kwargs['epsilon'],
                            activ_fun=activation,
                            graph_conv=kwargs['graph_conv'])

    def forward(self, x, edge_index):
        x = self.emb(x) if self.emb else x
        for conv in self.conv:
            x = conv(x, edge_index)

        x = self.readout(x)

        return x