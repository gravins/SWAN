import os
import torch
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_tar


def normalize(node_labels, graph_labels):
    # normalize labels
    max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
    max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
    for dset in node_labels.keys():
        node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
        graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]
    
    return node_labels, graph_labels

NODE_LVL_TASKS = ['sssp', 'ecc']
GRAPH_LVL_TASKS = ['diam']
TASKS = NODE_LVL_TASKS + GRAPH_LVL_TASKS

class GPPDataset(InMemoryDataset):
    url = 'https://github.com/gravins/Anti-SymmetricDGN/raw/refs/heads/main/graph_prop_pred/data.tar.gz'

    def __init__(self, root, name, split='train', pre_transform=None, transform=None):
        assert name in TASKS, f'{name} is not in {TASKS}'
        assert split in ['train', 'val', 'test']

        self.split = split
        self.name = name
        super().__init__(root, pre_transform=pre_transform, transform=transform)
        self.pre_transform = pre_transform
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_features(self) -> int:
        return self.data.x.size(1)
    
    @property
    def is_node_level_task(self) -> bool:
        return self.name in NODE_LVL_TASKS
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'data'

    @property
    def processed_file_names(self):
        return [f'{self.split}_{self.name}.pt']

    def download(self):
        targz = download_url(self.url, self.raw_dir)
        extract_tar(targz, self.raw_dir)
        os.unlink(targz)

    def process(self):
        (adj, features, node_labels, graph_labels) = torch.load(open(osp.join(self.raw_dir, self.raw_file_names, f'pna_dataset_25-35.pkl'),'rb'))

        node_labels, graph_labels = normalize(node_labels, graph_labels)

        data_list = []
        n_batches = len(adj[self.split])
        for batch_id in range(n_batches):
            n_samples_in_batch = len(adj[self.split][batch_id])
            for sample_id in range(n_samples_in_batch):
                
                a = adj[self.split][batch_id][sample_id]
                ft = features[self.split][batch_id][sample_id]
                nl = node_labels[self.split][batch_id][sample_id]
                gl = graph_labels[self.split][batch_id][sample_id]
                
                edge_index, _ = dense_to_sparse(a)
                
                if self.name == 'sssp':
                    y = nl[:, 2]
                elif self.name == 'ecc':
                    y = nl[:, 0]
                elif self.name == 'diam':
                    y = gl[1]
                else:
                    raise NotImplementedError()

                data = Data(x=ft, edge_index=edge_index, y=y)
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
