import torch
import numpy as np
import random
import pandas
import itertools

def set_seed(seed):
    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def update_csv(df, best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args, path):
    tmp = {
        'train_loss': best_train_loss, 
        'val_loss': best_val_loss, 
        'test_loss': best_test_loss,
        'convergence time (epochs)': best_epoch
    }
    tmp.update({f'model_{k}':v for k,v in model_params.items()})
    tmp.update(exp_args)
    df.append(tmp)
    pandas.DataFrame(df).to_csv(path, index=False)
    return df


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