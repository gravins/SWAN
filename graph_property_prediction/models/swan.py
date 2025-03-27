import torch

from typing import Optional
import math
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, to_undirected
from torch.nn.utils.parametrize import register_parametrization
from torch.nn import Module, Parameter, init, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.nn import MessagePassing, GCNConv, global_add_pool, global_max_pool, global_mean_pool

from collections import OrderedDict
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import os
import numpy as np

class AntiSymmetric(Module):
    r"""
    Anti-Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} - \mathbf{W}^T`
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu(diagonal=1)


class Symmetric(Module):
    r"""
    Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} + \mathbf{W}^T`
    """
    def __init__(self):
        super().__init__()

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu() + W.triu().T

    def right_inverse(self, W: torch.Tensor) -> torch.Tensor:
        return W.triu()


def get_adj(edge_index, edge_weight: Optional[torch.Tensor] = None,
            normalization: Optional[str] = 'sym',
            dtype: Optional[int] = None,
            num_nodes: Optional[int] = None):

    if normalization is not None:
        assert normalization in ['sym', 'rw', 'antisym']  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    edge_index,  edge_weight = to_undirected(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * 0.5 * (edge_weight[row] + edge_weight[col]) * deg_inv_sqrt[col]
    elif normalization == 'rw':
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    elif normalization == 'antisym':
        # Compute A_norm = (-D^{-1} A) - (-D^{-1} A).T
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
        edge_weight = edge_weight - edge_weight[col]
    else:
        print("PROBLEM")
    return edge_index, edge_weight


class ConstrainedConv(MessagePassing):
    def __init__(self, in_channels, edge_dim = 0, antisym = False, sym = False):
        super().__init__(aggr='add')
        assert not (antisym and sym)
        self.in_channels = in_channels
        self.antisym = antisym
        self.sym = sym
        self.lin = Linear(in_channels, in_channels, bias=False)
        self.lin_edge = Linear(edge_dim, in_channels, bias=False) if edge_dim > 0 else None
        if self.antisym: register_parametrization(self.lin, 'weight', AntiSymmetric())
        if self.sym: register_parametrization(self.lin, 'weight', Symmetric())

    def forward(self, x, edge_index=None, edge_weight=None):
        out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=self.lin(x))
        return out
    
    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        else:
            if self.lin_edge is None :
                if len(edge_weight.shape()) == 1:
                    return edge_weight.view(-1, 1) * x_j
                else:
                    ValueError("Node and edge feature dimensionalities do not match.")
            else:
                return x_j + self.lin_edge(edge_weight)

    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels:{self.in_channels},antisym:{self.antisym},sym:{self.sym})'
    

conv_names = ['AntiSymNaiveAggr', 'BoundedGCNConv', 'BoundedNaiveAggr']
class SWANConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 gamma: float = 0.1, 
                 epsilon : float = 0.1, 
                 beta: float = 0.5,
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 graph_conv: str = 'AntiSymNaiveAggr',
                 attention: bool = False, # if true then SWAN_learn is used
                 bias: bool = True) -> None:

        super().__init__(aggr = 'add')
        self.W = Parameter(torch.empty((in_channels, in_channels)))
        self.bias = Parameter(torch.empty(in_channels)) if bias else None

        edge_dim = in_channels if attention else 0
        if graph_conv == 'AntiSymNaiveAggr':
            self.conv = ConstrainedConv(in_channels, antisym=True,
                                        edge_dim=edge_dim) # Global and Local Non-Dissipative
        elif graph_conv == 'BoundedGCNConv':
            self.gnn = GCNConv(in_channels, in_channels, bias=False) # Bounded non-dissipative
        elif graph_conv == 'BoundedNaiveAggr':
            self.conv = ConstrainedConv(in_channels) # Bounded non-dissipative
        else:
            NotImplementedError(f'{graph_conv} not implemented. {graph_conv} is not in {conv_names}')

        self.antisym_conv = ConstrainedConv(in_channels, sym=True,
                                            edge_dim=edge_dim) # non-dissipative over space

        if attention:
            assert graph_conv != 'BoundedGCNConv'
            self.edge_emb2 = torch.nn.Linear(2 * in_channels, out_features=in_channels)
            self.edge_emb3 = torch.nn.Linear(in_channels, out_features=in_channels)
            self.edge_bn = torch.nn.BatchNorm1d(in_channels)
        self.attention = attention

        self.graph_conv = graph_conv
        self.in_channels = in_channels
        self.num_iters = num_iters
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)
        self.activ_fun=activ_fun

        self.reset_parameters()
    
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.attention:
            edge_feats = torch.cat([x[edge_index[0, :].long()], x[edge_index[1, :].long()]], dim=-1)
            if edge_weight is not None:
                edge_feats += edge_weight
            edge_weight = (self.edge_bn(self.edge_emb2(edge_feats)))
            edge_weight = torch.nn.functional.relu(self.edge_emb3(torch.nn.functional.leaky_relu(edge_weight, 0.2)))

        self.antisymmetric_W = self.W - self.W.T - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        (self.antisym_edge_index, 
         self.antisym_edge_weight) = get_adj(edge_index, edge_weight=edge_weight, normalization='antisym')
        
        if self.graph_conv == 'AntiSymNaiveAggr':
            (self.sym_edge_index, 
             self.sym_edge_weight) = get_adj(edge_index, edge_weight=edge_weight, normalization='sym')
        
        for _ in range(self.num_iters):
            if self.graph_conv == 'AntiSymNaiveAggr':
                neigh_x = self.conv(x, edge_index=self.sym_edge_index, edge_weight=self.sym_edge_weight)
            else:
                neigh_x = self.conv(x, edge_index=edge_index, edge_weight=edge_weight)
            antisym_neigh_x = self.antisym_conv(x, edge_index=self.antisym_edge_index, edge_weight=self.antisym_edge_weight)

            conv = x @ self.antisymmetric_W.T + neigh_x + self.beta * antisym_neigh_x

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



class SWAN(Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_layers,
                 epsilon,
                 gamma=0.1,
                 beta=0.1,
                 activ_fun='tanh',
                 graph_conv: str = 'AntiSymNaiveAggr',
                 attention: bool = False, # if true then SWAN_learn is used
                 node_level_task=False,
                 train_weights: bool = True, 
                 weight_sharing: bool = True,
                 bias: bool = True) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.activ_fun = activ_fun
        self.bias = bias
        self.train_weights = train_weights
        self.weight_sharing = weight_sharing
        self.attention = attention

        self.emb = Linear(self.input_dim, self.hidden_dim)

        self.convs = ModuleList()
        for _ in range(1 if self.weight_sharing else self.num_layers):
            self.convs.append(SWANConv(
                in_channels=self.hidden_dim,
                num_iters=self.num_layers if weight_sharing else 1,
                gamma=self.gamma,
                epsilon = self.epsilon,
                beta=self.beta,
                attention=self.attention,
                activ_fun=self.activ_fun,
                graph_conv=graph_conv,
                bias=self.bias
            ))
            
        if not train_weights:
            #for param in self.enc.parameters():
            #    param.requires_grad = False
            for param in self.conv.parameters():
                param.requires_grad = False

        self.node_level_task = node_level_task 
        # Original code from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023
        # https://github.com/gravins/Anti-SymmetricDGN/blob/main/graph_prop_pred/models/dgn_GraphProp.py
        if self.node_level_task:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(self.hidden_dim, self.hidden_dim // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(self.hidden_dim // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(self.hidden_dim * 3, (self.hidden_dim * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((self.hidden_dim * 3) // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            
        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)
        return x
