import torch
import ray
from graph_transfer_data import GraphTransferDataset
from torch_geometric.loader import DataLoader
import numpy as np
import os


def optimizer_to(optim, device):
    # Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        
                        
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = torch.nn.functional.mse_loss(out, data.y)

    loss.backward()
    optimizer.step()


def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.mse_loss(out, data.y)

    return loss.detach().cpu(), out.detach().cpu()


@ray.remote(num_cpus=1, num_gpus=1/8)
def run_exp(data_name, distance, channels, model_params, args):
    return run_single_exp(data_name, distance, channels, model_params, args)


def run_single_exp(data_name, distance, channels, model_params, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_transform = None
    train_data = GraphTransferDataset(root=args['data_path'], name=data_name, distance=distance, split='train', pre_transform=pre_transform)
    valid_data = GraphTransferDataset(root=args['data_path'], name=data_name, distance=distance, split='val', pre_transform=pre_transform)
    test_data = GraphTransferDataset(root=args['data_path'], name=data_name, distance=distance, split='test', pre_transform=pre_transform)

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=len(train_data) if args['batch']==None else args['batch'])

    valid_loader = DataLoader(valid_data,
                              shuffle=False,
                              batch_size=len(valid_data) if args['batch']==None else args['batch'])

    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=len(test_data) if args['batch']==None else args['batch'])
    
    model = args['model_instance'](**model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_epoch = 0
    best_train_loss = np.inf
    best_val_loss = None
    best_test_loss = np.inf
    
    # Load previuos ckpt if exists
    ckpt_path = args['ckpt_path']
    if os.path.exists(ckpt_path):
        # Load the existing checkpoint
        print(f'Loading {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
    
        train_ended = ckpt['train_ended']
        if train_ended:
            return ckpt['tr_loss'], ckpt['vl_loss'], ckpt['ts_loss'], model_params, args

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        optimizer_to(optimizer, device) # Map the optimizer to the current device

        best_epoch = ckpt['best_epoch']
        best_train_loss = ckpt['tr_loss']
        best_val_loss = ckpt['vl_loss']
        best_test_loss = ckpt['ts_loss']

    model.to(device)

    for epoch in range(best_epoch, args['epochs']):
        train_loss, valid_loss, test_loss = [], [], []

        #train
        for batch in train_loader:
            train(model, optimizer, batch.to(device))
            train_loss_batch, out_batch = test(model, batch.to(device))
            train_loss.append(train_loss_batch)

        #valid
        for batch in valid_loader:
            valid_losses_batch, out_batch = test(model, batch.to(device))
            valid_loss.append(valid_losses_batch)

        
        #eval test
        for batch in test_loader:
            test_losses_batch, out_batch = test(model, batch.to(device))
            test_loss.append(test_losses_batch) 
        ts_loss = np.mean(test_loss)
        
        tr_loss, vl_loss = np.mean(train_loss), np.mean(valid_loss)
        if best_val_loss is None or vl_loss <= best_val_loss:
            #eval test
            for batch in test_loader:
                test_losses_batch, out_batch = test(model, batch.to(device))
                test_loss.append(test_losses_batch) 
            ts_loss = np.mean(test_loss)

            best_epoch = epoch
            best_train_loss = tr_loss
            best_val_loss = vl_loss
            best_test_loss = ts_loss

            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tr_loss': tr_loss, 
                    'vl_loss': vl_loss, 
                    'ts_loss': ts_loss,
                    'train_ended': False,
                    'best_valid_loss': valid_loss,
                    'best_epoch': epoch
            }, ckpt_path)

        if args['verbose'] and epoch % 100 == 0:
            print("TR loss:", tr_loss, "VL loss:", vl_loss, "TS loss:", ts_loss, "Epoch:", epoch)
            print('<y_pred, y_true>')

        if epoch - best_epoch > args['early_stopping']:
            break
    
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt['train_ended'] = True
    best_epoch = ckpt['best_epoch']
    torch.save(ckpt, ckpt_path)

    return best_train_loss, best_val_loss, best_test_loss, best_epoch, model_params, args
