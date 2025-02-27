import itertools
from models import *

def cartesian_product(params):
    # Given a dictionary where for each key is associated a lists of values, the function compute cartesian product
    # of all values. 
    # Example:
    #  Input:  params = {"n_layer": [1,2], "bias": [True, False] }
    #  Output: {"n_layer": [1], "bias": [True]}
    #          {"n_layer": [1], "bias": [False]}
    #          {"n_layer": [2], "bias": [True]}
    #          {"n_layer": [2], "bias": [False]}
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def config_SWAN(num_features, num_classes, *args, **kwargs):
    # Parameters for the grid search
    grid = {
        'hidden_dim': [30, 20, 10], # Original hyperparamerers from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023 
        'num_layers': [20, 10, 5, 1], # Original hyperparamerers from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023
        'weight_sharing': [True, False],
        'epsilon': [1., 1e-1, 1e-2],
        'gamma': [1, 0.1],
        'beta': [2, 1, 0.5, 0.1, -0.5, -1],
        'attention': [True, False],
        'graph_conv': ['AntiSymNaiveAggr', 'BoundedGCNConv', 'BoundedNaiveAggr']
    }
    
    # Iterate through the cartesian product of the grid
    for conf in cartesian_product(grid):
        if conf['graph_conv'] == 'BoundedGCNConv' and conf['attention']:
            continue
        fixed_hyperparams =  {
            'model': {
                'input_dim': num_features,
                'output_dim': num_classes,
                'activ_fun': 'tanh',
                'bias': True,
                'num_layers': 1,
            },
            'optim': { # Original hyperparamerers from Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. ICLR 2023
                'lr': 0.003,
                'weight_decay': 1e-6
            }
        }
        
        fixed_hyperparams['model'].update(conf)
        yield fixed_hyperparams



swan_ = lambda num_features, num_classes, task: config_SWAN(num_features, num_classes, task)

CONFIGS = {
    'SWAN': (swan_, SWAN)
}


