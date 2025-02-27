import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import ray
import time
import tqdm
import pandas
import argparse
import datetime
import numpy as np
from graph_transfer_data import distributions, GraphTransferDataset
from train import run_exp, run_single_exp
from utils import update_csv
from conf import MODELS

if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', type=str, choices=list(MODELS.keys()),
                        default=list(MODELS.keys())[0])
    # Optimisation params
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--early_stopping', type=int, default=100)

    # Experiment parameters
    parser.add_argument('--distance', type=int, default=64)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--root', type=str,  default='./oversquashing_exps')
    parser.add_argument('--parallelism', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=None)

    parser.add_argument('--ncpus', help="Num. CPUs per task", type=int, default=1)
    parser.add_argument('--ngpus', help="Num. GPUs per task", type=float, default=1.)
    args = parser.parse_args()

    seeds = [100,200,300,400]
    
    if not args.debug:
        ray.init(address='local')  # local ray initialization

    args.root = os.path.abspath(args.root)

    bres = {f'best run {data}':np.inf for data in distributions}
    for data in distributions:
        exp_list = []
        model_name = args.m

        data_path = os.path.join(args.root, 'data')
        # Create dataset if it does not exist
        pre_transform = None
        tr = GraphTransferDataset(root=data_path, name=data, distance=args.distance, split='train', pre_transform=pre_transform)
        vl = GraphTransferDataset(root=data_path, name=data, distance=args.distance, split='val', pre_transform=pre_transform)
        ts = GraphTransferDataset(root=data_path, name=data, distance=args.distance, split='test', pre_transform=pre_transform)

        model_path = os.path.join(args.root, 'results', f'{data}_{args.distance}',
                                  f'{model_name}_{pre_transform.__name__}' if pre_transform is not None else model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(os.path.join(model_path, 'ckpt'))

        model_instance, getconf = MODELS[model_name]
        
        i = 0
        for conf in getconf(tr.num_features, args.distance):
            for seed in seeds:
                exp_args = {
                    'data_path': data_path,
                    'model': model_name,
                    'model_instance': model_instance,
                    'ckpt_path': os.path.join(model_path, 'ckpt', f'{model_name}_id{i}_seed{seed}.pt'),
                    'epochs': args.epochs,
                    'verbose': args.verbose,
                    'early_stopping': args.early_stopping,
                    'conf_id': f'{model_name}_{i}',
                    'seed': seed
                }
                exp_args.update(vars(args))

                exp_list.append({
                    'data_name': data, 
                    'distance': args.distance, 
                    'channels': tr.num_features, 
                    'model_params': conf, 
                    'args': exp_args
                })
            i += 1         
        del tr, vl, ts

        ray_ids = []
        df = []
        pbar = tqdm.tqdm(total=len(exp_list))
        partial_results_path = os.path.join(model_path, f'partial_results_{model_name}.csv')
        for exp in exp_list:
            if args.debug:
                (best_train_loss, best_val_loss, best_test_loss, best_epoch,
                    model_params, exp_args) = run_single_exp(**exp)
                df = update_csv(df, best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args,
                           path = partial_results_path)
                if np.isinf(bres[f'best run {data}']) or best_val_loss < bres[f'best run {data}']:
                    bres[f'best run {data}'] = best_test_loss
                    pbar.set_postfix(bres) 
                pbar.update(1)
            else:
                opt = {
                    'num_cpus': args.ncpus, 
                    'num_gpus': args.ngpus
                }
                ray_ids.append(run_exp.options(**opt).remote(**exp))
                while len(ray_ids) >= args.parallelism:
                    done_id, ray_ids = ray.wait(ray_ids)
                    best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args = ray.get(done_id[0])
                    df = update_csv(df, best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args,
                           path = partial_results_path)
                    if np.isinf(bres[f'best run {data}']) or best_val_loss < bres[f'best run {data}']:
                        bres[f'best run {data}'] = best_test_loss
                        pbar.set_postfix(bres) 
                    pbar.update(1)

        while len(ray_ids):
            done_id, ray_ids = ray.wait(ray_ids)
            best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args = ray.get(done_id[0])
            df = update_csv(df, best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, exp_args,
                           path = partial_results_path)
            if np.isinf(bres[f'best run {data}']) or [f'best run {data}']:
                bres[f'best run {data}'] = best_test_loss
                pbar.set_postfix(bres) 
            pbar.update(1)

        df = pandas.read_csv(partial_results_path)

        rows = []
        for model_id, gdf in df.groupby('conf_id'):
            gdf['mean_train_loss'] = gdf['train_loss'].mean()
            gdf['std_train_loss'] = gdf['train_loss'].std()
            gdf['mean_val_loss'] = gdf['val_loss'].mean()
            gdf['std_val_loss'] = gdf['val_loss'].std()
            gdf['mean_test_loss'] = gdf['test_loss'].mean()
            gdf['std_test_loss'] = gdf['test_loss'].std()
            gdf['avg convergence time (epochs)'] = gdf['convergence time (epochs)'].mean()
            gdf['std convergence time (epochs)'] = gdf['convergence time (epochs)'].std()
            cols = [c for c in gdf.columns if c not in ['train_loss', 'val_loss', 'test_loss', 'convergence time (epochs)']]            
            rows.append(gdf[cols].iloc[0]) # since, we copied the mean/std values to all the runs we can just take one row

        fdf = pandas.concat(rows, axis=1).T
        fdf = fdf.sort_values('mean_val_loss', ascending=True)
        fdf.to_csv(os.path.join(model_path, f'final_results_{model_name}.csv'))
        print(data, '\n\t top 10 resulst:\n', fdf[['conf_id', 'mean_test_loss', 'std_test_loss', 'mean_val_loss', 'std_val_loss', 'mean_train_loss', 'std_train_loss', 'avg convergence time (epochs)', 'std convergence time (epochs)']])

    elapsed = time.time() - t0
    if not args.debug:
        ray.shutdown()
    print(datetime.timedelta(seconds=elapsed))
