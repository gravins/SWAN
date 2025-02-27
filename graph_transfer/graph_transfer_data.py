import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from utils import set_seed
import os.path as osp

ring = 'ring'
crossedring = 'crossed-ring'
ring_types = [ring, crossedring]
line = 'line'
cliquepath = 'cliquepath'
line_types = [line, cliquepath]
distributions = [line, ring, crossedring] ##ring_types + line_types


def line_graph(distance, channels=1):
    assert distance > 1

    n_nodes = distance

    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes-1):
        A[i, i+1] = 1

    # add self loops
    A = A + torch.eye(n_nodes)

    # make the graph undirected
    A = A.triu()
    A = A + A.t()

    edge_index = A.nonzero().T

    x = torch.rand(n_nodes, channels) 
    x[0, :] = 1
    x[-1, :] = 0

    y = x.clone()
    y[0, :] = 0
    y[-1, :] = 1

    return Data(x=x, edge_index=edge_index, y=y)


def cliquepath_transfer_graph(distance, channels=1):    
    assert distance > 3

    # d = n/2 + 1
    n_nodes = (distance - 1) / 2

    if n_nodes <= 1: raise ValueError("Minimum of two nodes required")    
    # Initialize node features. The first node gets 0s, while the last gets the target label
    
    x = torch.rand(n_nodes, channels)

    x[0, :] = 0
    x[n_nodes - 1, :] = 1
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    # Construct a clique for the first half of the nodes, 
    # where each node is connected to every other node except itself
    for i in range(n_nodes // 2):
        for j in range(n_nodes // 2):
            if i == j:  # Skip self-loops
                continue
            edge_index.append([i, j])
            edge_index.append([j, i])

    # Construct a path (a sequence of connected nodes) for the second half of the nodes
    for i in range(n_nodes // 2, n_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    # Connect the last node of the clique to the first node of the path
    edge_index.append([n_nodes // 2 - 1, n_nodes // 2])
    edge_index.append([n_nodes // 2, n_nodes // 2 - 1])

    # Convert the edge index list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    y = x.clone()
    y[0, :] = 0
    y[n_nodes - 1, :] = 1

    return Data(x=x, edge_index=edge_index, y=y)


def ring_transfer_graph(distance, channels, add_crosses: bool):
    assert distance > 1
    n_nodes = distance * 2

    assert n_nodes > 1, ValueError("Minimum of two nodes required")
    # Determine the node directly opposite to the source (node 0) in the ring
    opposite_node = n_nodes // 2

    # Initialise feature matrix with a uniform feature. 
    x = torch.rand(n_nodes, channels)

    # Set feature of the source node to 0 and the opposite node to the target label
    x[0, :] = 1
    x[opposite_node, :] = 0

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(n_nodes-1):
        # Regular connections that make the ring
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
        
        # Conditionally add cross edges, if desired
        if add_crosses and i < opposite_node:
            # Add edges from a node to its direct opposite
            edge_index.append([i, n_nodes - 1 - i])
            edge_index.append([n_nodes - 1 - i, i])

            # Extra logic for ensuring additional "cross" edges in some conditions
            if n_nodes + 1 - i < n_nodes:
                edge_index.append([i, n_nodes + 1 - i])
                edge_index.append([n_nodes + 1 - i, i])

    # Close the ring by connecting the last and the first nodes
    edge_index.append([0, n_nodes - 1])
    edge_index.append([n_nodes - 1, 0])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    y = x.clone()
    y[0, :] = 0
    y[opposite_node, :] = 1

    return Data(x=x, edge_index=edge_index, y=y)


class GraphTransferDataset(InMemoryDataset):
    def __init__(self, root, name, distance, split='train', pre_transform=None, transform=None):
        assert name in distributions, f'{name} is not in {distributions}'
        assert split in ['train', 'val', 'test']

        self.split = split
        self.name = name
        self.distance = distance
        self.pre_transform = pre_transform
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        #self.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def num_features(self) -> int:
        return 1
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.name}_{self.distance}', f'pre_transform_{self.pre_transform.__name__ if self.pre_transform is not None else "None"}')

    @property
    def processed_file_names(self):
        return [f'{self.split}_{self.name}_{self.distance}_pre_transform_{self.pre_transform.__name__ if self.pre_transform is not None else None}.pt']

    def process(self):
        preprocessed = osp.join(self.processed_dir, f'{self.name}_{self.distance}.pt')
        data_list = []
        if not osp.exists(preprocessed):
            seed = 1234 if self.split == 'train' else 2345 if self.split == 'valid' else 3456
            num_graphs = 1000 if self.split == 'train' else 100
            if seed is not None: set_seed(seed)
            untransformed_data = []
            for _ in range(num_graphs):
                if self.name in ring_types:
                    # ring/crossed-ring
                    g = ring_transfer_graph(distance = self.distance, 
                                            channels = self.num_features,
                                            add_crosses = self.name==crossedring)
                else:
                    # line:
                    g = line_graph(distance=self.distance, 
                                   channels=self.num_features)
                
                untransformed_data.append(g)
        else:
            tr, vl, ts = torch.load(preprocessed)
            untransformed_data = tr if self.split == 'train' else vl if self.split == 'valid' else ts

        for data in untransformed_data:
            data = (data if self.pre_transform is None 
                    else self.pre_transform(data))
            data_list.append(data)

        #self.save(data_list, self.processed_paths[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])