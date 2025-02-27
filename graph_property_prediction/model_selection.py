import torch

import os
import ray
import tqdm
import pandas as pd
from conf import CONFIGS
from utils import get_dataset
from train_GraphProp import train_val_pipeline_GraphProp
from utils.io import dump, join, create_if_not_exist
from typing import Optional
from utils.gpp_dataset import NODE_LVL_TASKS

def model_selection(model_name: str,
                    early_stopping_patience: Optional[int] = None,
                    epochs: int = 1000,
                    task = None,
                    data_dir: str = '.',
                    exp_dir: str = '.',
                    num_cpus=1,
                    num_gpus=0.):
    """
    Perform a model selection phase through standard validation or k-fold model selection.
    All the results are saved into a DataFrame and the best configuration is returned.
    """

    assert ray.is_initialized() == True, "Ray is not initialized"
    data_dir = os.path.abspath(data_dir) # ray wants absolute paths
    exp_dir = os.path.abspath(exp_dir)

    assert not os.path.exists(join(exp_dir, 'results.csv')), 'The file results.csv already exists.'
    
    # Download data once for all configurations
    data_train, data_valid, data_test, num_features, num_classes = get_dataset(root=data_dir, task=task)
    del data_train, data_valid, data_test

    # Create the checkpoint directory
    checkpoint_dir = join(exp_dir, 'checkpoints')
    create_if_not_exist(checkpoint_dir)

    config_fun, model = CONFIGS[model_name]
    ray_ids = []
    ids_to_configs = {}
    
    batch_size = 128 #512
    seeds = [41, 95, 12, 35]
        
    for conf_id, conf in enumerate(config_fun(num_features, num_classes, task)):
            conf.update({
                'exp':{'conf_id': conf_id,
                       'task': task,
                       'epochs': epochs,
                       'patience': early_stopping_patience,
                       'batch_size': batch_size,
                       'seeds': seeds,
                       }
            })
            conf['model'].update({
                'input_dim': num_features,
                'output_dim': num_classes,
                'node_level_task': task in NODE_LVL_TASKS
            })
            
            checkpoint_path = join(checkpoint_dir, f'conf_id_{conf_id}.pth')
            ray_ids.append( 
                    train_val_pipeline_GraphProp.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(model, conf, data_dir, early_stopping_patience, checkpoint_path)
                )
            
            ids_to_configs[ray_ids[-1]] = conf.copy()
         
    df = []
    final_json = []
    best_score = None
    # Wait and collect results
    tqdm_ids = tqdm.tqdm(total=len(ray_ids))
    while len(ray_ids):
        done_id, ray_ids = ray.wait(ray_ids)
        id_ = done_id[0]
        res = ray.get(id_)

        tqdm_ids.update(1)

        conf = ids_to_configs[id_]
        result = {} #'ray_id': id_}
        for key_name, values in conf.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    result[f'{key_name}_{k}'] = v
            else:
                result[key_name] = values

        avg = res['avg_res']
        result.update(avg)
        df.append(result)
        pd.DataFrame(df).sort_values('avg best_val_score').to_csv(join(exp_dir, 'partial_results.csv'), index=False)

        if best_score is None or avg['avg best_val_score'] < best_score:
            best_score = avg['avg best_val_score']
            tqdm_ids.set_postfix(best_train_loss = avg['avg best_train_loss'],
                            best_val_loss = avg['avg best_val_loss'],
                            best_test_loss = avg['avg best_test_loss'],
                            best_train_log10_MSE = avg['avg best_train_score'],
                            best_val_log10_MSE = avg['avg best_val_score'],
                            best_test_log10_MSE = avg['avg best_test_score'])    

        final_json.append(res)
    
    json_path = join(exp_dir, 'complete_results.json')
    dump(final_json, json_path)

    df = pd.DataFrame(df)
    csv_path = join(exp_dir, 'results.csv')
    df.to_csv(csv_path, index=False)

    final_json.sort(key=lambda x: x['avg_res']['avg best_val_score']) # smaller values are the best
    df = df.sort_values('avg best_val_score', ascending=True, ignore_index=True)

    dump(final_json, json_path)
    df.to_csv(csv_path, index=False)
    
    return final_json[0]
