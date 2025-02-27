from models import *
from utils import cartesian_product
 

def get_GNN_conf(in_channels, distance):
    # GCN, GAT, GraphSAGE, GIN, GPS
    grid = {
        'hidden_channels':[64],
        'num_layers': [distance],
        'activation':['tanh'],
    }
    for params in cartesian_product(grid):
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_ADGN_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers': [1],
        'activation':['tanh'],
        'num_iters': [distance],
        'gamma': [0.1, 0.2],
        'epsilon': [0.6, 0.2, 0.5, 0.1],
        'graph_conv': ['NaiveAggr', 'GCNConv']
    }
    for params in cartesian_product(grid):
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_SWAN_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers': [1],
        'activation':['tanh'],
        'num_iters': [distance],
        'gamma': [0.1, 0.2],
        'epsilon': [0.6, 0.2, 0.5, 0.1],
        'beta': [1., -1., 0.1, 0.01], #0.01, 0]:,
        'graph_conv': ['AntiSymNaiveAggr', 'BoundedGCNConv', 'BoundedNaiveAggr'],
        'attention': [False, True]
    }
    for params in cartesian_product(grid):
        if params['attn'] and params['conv'] == 'BoundedGCNConv':
            continue
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params



MODELS = {
    'gin' : (GIN_Model, get_GNN_conf),
    'gcn' : (GCN_Model, get_GNN_conf),
    'gat' : (GAT_Model, get_GNN_conf),
    'sage': (SAGE_Model, get_GNN_conf),
    'gps': (GPS_Model, get_GNN_conf),
    'adgn': (ADGN_Model, get_ADGN_conf),
    'swan': (SWAN_Model, get_SWAN_conf)
}