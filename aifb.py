
import os
import numpy as np
import scipy.sparse as sp
import pdb
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
    
    def __init__(self, dataset_name="aifb", dataset_path="/home/msabrishami/.dgl/aifb"):
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
        self.num_node = all_edges['n'].item()
        self.edge_list = all_edges['edges']
        self.num_rel = all_edges['nrel'].item()

        self.labels = _load_sparse_csr(labels_file)
        self.labeled_nodes_idx = list(self.labels.nonzero()[0])
        
        self.train_idx = np.load(train_idx_file)
        self.test_idx = np.load(test_idx_file)

        print('\t- Number of nodes: ', self.num_node)
        print('\t- Number of edges: ', len(self.edge_list))
        print('\t- Number of relations: ', self.num_rel)
        print('\t- Number of classes: ', self.labels.shape[1])

    
    def get_pytorch_geometric_dataset(self):
        dataset = Data
