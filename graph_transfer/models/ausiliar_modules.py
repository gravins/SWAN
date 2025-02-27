import torch

from torch.nn import Linear, Module
from torch_geometric.nn import MessagePassing
from torch.nn.utils.parametrize import register_parametrization
from typing import Optional
from torch.nn import Module

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


class ConstrainedConv(MessagePassing):
    def __init__(self, in_channels, antisym = False, sym = False, edge_dim = 0):
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
            if self.lin_edge is None:
                if len(edge_weight.shape()) == 1:
                    return edge_weight.view(-1, 1) * x_j
                else:
                    ValueError("Node and edge feature dimensionalities do not match.")
            else:
                return x_j + self.lin_edge(edge_weight)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels:{self.in_channels},antisym:{self.antisym},sym:{self.sym})'
    

class NaiveAggr(ConstrainedConv):
    r"""
    Simple graph convolution which compute a transformation of 
    neighboring nodes:  sum_{j \in N(u)} Vx_j
    """
    def __init__(self, in_channels):
        super().__init__(in_channels, antisym = False, sym = False)
