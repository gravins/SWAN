import torch

from torch.nn import Parameter, init
from torch_geometric.nn import MessagePassing, GCNConv
from typing import Optional
import math
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops, to_undirected
from models.ausiliar_modules import ConstrainedConv
from models.gnn_model import BasicModel


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
            edge_weight = self.edge_emb3(torch.nn.functional.leaky_relu(edge_weight, 0.2))

        self.antisymmetric_W = self.W - self.W.T - self.gamma * torch.eye(self.in_channels, device=self.W.device)
        (self.antisym_edge_index, 
         self.antisym_edge_weight) = get_adj(edge_index, edge_weight=edge_weight, normalization='antisym')
        
        if self.graph_conv == 'AntiSymNaiveAggr':
            (self.sym_edge_index, 
             self.sym_edge_weight) = get_adj(edge_index, edge_weight=edge_weight, normalization='sym')
        
        for i in range(self.num_iters):
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




class SWAN_Model(BasicModel):
    def init_conv(self, in_channels: int, out_channels: int, activation:str, **kwargs) -> MessagePassing:
            return SWANConv(in_channels=in_channels,
                            num_iters=kwargs['num_iters'],
                            gamma=kwargs['gamma'],
                            epsilon=kwargs['epsilon'],
                            beta=kwargs['beta'],
                            activ_fun=activation,
                            graph_conv=kwargs['graph_conv'],
                            attention=kwargs['attention'])

    def forward(self, x, edge_index):
        x = self.emb(x) if self.emb else x
        for conv in self.conv:
            x = conv(x, edge_index)

        x = self.readout(x)

        return x