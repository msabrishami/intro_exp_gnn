
import os
import numpy as np
import scipy.sparse as sp
import pdb
import torch
from torch_geometric.data import InMemoryDataset, Data

''' More simple approach:
    import dgl.contrib.data.knowledge_graph as knwlgrh
    temp = knwlgrh.load_entity("aifb", 3, False)
'''

def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors



def _bfs_relational(adj, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(next_lvl)



def _load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape'], dtype=np.float32)


class AIFB(object):
    ''' based based on DGL library.
    '''
    
    def __init__(self, dataset_name="aifb", dataset_path=None):
        self.name = dataset_name
        self.path = dataset_path

    def download_library():
        print("Not implemented yet")
        return True

    def load_data(self):
        ''' based on dgl/contrib/data/knowledge_graph._load_data
        information about the files: (should be moved to download method)
        there are 4 files:
            
        format: numpy.lib.npyio.NpzFile
            description: NpzFile is used to load files in the NumPy .npz data archive format. 
            It assumes that files in the archive have a .npy extension, other files are ignored.
            NpzFile.files(): List of all files in the archive with a .npy extension.

        - edges.npz ['edges', 'n', 'nrel']
            edges: 3 dim, I don't know what is the 3rd dimension (90 unique values)

        - labels.npz ['data', 'indices', 'indptr', 'shape']
            data: ones(176)
            indices: 176x[a number in range 0-3]
            indptr: 8286, with 177 unique values between 0-176
            shape: (8285, 4)


        '''
        graph_file = os.path.join(self.path, '{}_stripped.nt.gz'.format(self.name))
        task_file = os.path.join(self.path, 'completeDataset.tsv')
        train_file = os.path.join(self.path, 'trainingSet.tsv')
        test_file = os.path.join(self.path, 'testSet.tsv')

        # label_header = 'label_affiliation'
        # nodes_header = 'person'

        edge_file       = os.path.join(self.path, 'edges.npz')
        labels_file     = os.path.join(self.path, 'labels.npz')
        train_idx_file  = os.path.join(self.path, 'train_idx.npy')
        test_idx_file   = os.path.join(self.path, 'test_idx.npy')
        
        all_edges = np.load(edge_file)
        self.num_nodes = all_edges['n'].item()
        self.edge_list = all_edges['edges']
        self.num_rel = all_edges['nrel'].item()

        self.labels = _load_sparse_csr(labels_file)
        self.labeled_nodes_idx = list(self.labels.nonzero()[0])
        pdb.set_trace()
        
        self.train_idx = np.load(train_idx_file)
        self.test_idx = np.load(test_idx_file)

        print('\t- Raw data loaded from ' + self.path)
        print('\t- Num. of nodes: ', self.num_nodes)
        print('\t- Num. of edges: ', len(self.edge_list))
        print('\t- Num. of relations: ', self.num_rel)
        print('\t- Num. of classes: ', self.labels.shape[1])
        print('\t- Num. of train: ', len(self.train_idx))
        print('\t- Num. of test: ', len(self.test_idx))
    
    def preprocess(self, bfs_level=3, relabel=False):
        
        if bfs_level > 0:
            print("Removing nodes that are more than {} hops away".format(bfs_level))
            row, col, edge_type = self.edge_list.transpose()
            A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
            bfs_generator = _bfs_relational(A, self.labeled_nodes_idx)
            lvls = list()
            lvls.append(set(self.labeled_nodes_idx))
            for _ in range(bfs_level):
                lvls.append(next(bfs_generator))
            to_delete = list(set(range(self.num_nodes)) - set.union(*lvls))
            eid_to_delete = np.isin(row, to_delete) + np.isin(col, to_delete)
            eid_to_keep = np.logical_not(eid_to_delete)
            self.edge_src = row[eid_to_keep]
            self.edge_dst = col[eid_to_keep]
            self.edge_type = edge_type[eid_to_keep]

            if relabel:
                uniq_nodes, edges = np.unique((self.edge_src, self.edge_dst), return_inverse=True)
                self.edge_src, self.edge_dst = np.reshape(edges, (2, -1))
                node_map = np.zeros(self.num_nodes, dtype=int)
                self.num_nodes = len(uniq_nodes)
                node_map[uniq_nodes] = np.arange(self.num_nodes)
                self.labels = self.labels[uniq_nodes]
                self.train_idx = node_map[self.train_idx]
                self.test_idx = node_map[self.test_idx]
                print("{} nodes left".format(self.num_nodes))
        else:
            self.edge_src, self.edge_dst, self.edge_type = edges.transpose()

        # normalize by dst degree
        _, inverse_index, count = np.unique((self.edge_dst, self.edge_type), axis=1, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        self.edge_norm = np.ones(len(self.edge_dst), dtype=np.float32) / degrees.astype(np.float32)

        # convert to pytorch label format
        self.num_classes = self.labels.shape[1]
        self.labels = np.argmax(self.labels, axis=1)


class AIFB_pyg(InMemoryDataset):
    ''' is_directed method is not working
    '''

    def __init__(self, dataset_path, transform=None):
        super(AIFB_pyg, self).__init__(".", transform, None, None)
        self.dataset_path = dataset_path
        ds = AIFB(dataset_path=dataset_path)
        ds.load_data()
        pdb.set_trace()
        ds.preprocess()
        src_tensor = torch.from_numpy(ds.edge_src.astype(np.int64)).to(torch.long)
        dst_tensor = torch.from_numpy(ds.edge_dst.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([src_tensor, dst_tensor], dim=0)
        data = Data(edge_index=edge_index)

        data.num_nodes = ds.num_nodes   # need to be checked after preprocess
        # data.x = torch.eye(data.num_nodes, dtype=torch.float)
        data.x = torch.ones(data.num_nodes, 1)
        data.y = torch.tensor(ds.labels).squeeze()
        data.train_mask = torch.tensor([False] * data.num_nodes)
        data.test_mask = torch.tensor([False] * data.num_nodes)
        data.train_mask[ds.train_idx] = True
        data.test_mask[ds.test_idx] = True
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
   
    def info(self):
       print("Num. of nodes: {}".format(self.data.num_nodes[0]))
       print("Num. of edges: {}".format(self.data.num_edges))
       print("Num. of train samples: {}".format(sum(self.data.train_mask).item()))
       print("Num. of test samples: {}".format(sum(self.data.test_mask).item()))
       print("Num. of samples per class in training dataset:", end=" ")
       for i in range(4): print(sum(self.data.y[self.data.train_mask] == i).item(), end=" ")
       print()
       print("Num. of samples per class in test dataset:", end=" ")
       for i in range(4): print(sum(self.data.y[self.data.test_mask] == i).item(), end=" ")


