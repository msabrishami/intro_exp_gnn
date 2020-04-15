
import os
import numpy as np
import scipy.sparse as sp


''' More simple approach:
    import dgl.contrib.data.knowledge_graph as knwlgrh
    temp = knwlgrh.load_entity("aifb", 3, False)
'''


def _load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
            shape=loader['shape'], dtype=np.float32)

def _load_data(dataset_str='aifb', dataset_path=None):
    """

    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print("\t>> =======================================================================")
    print("\t>> This is a stand alone function _load_data with no knowledge graph class")
    print('\t>> Loading dataset', dataset_str, dataset_path)
    graph_file = os.path.join(dataset_path, '{}_stripped.nt.gz'.format(dataset_str))
    task_file = os.path.join(dataset_path, 'completeDataset.tsv')
    train_file = os.path.join(dataset_path, 'trainingSet.tsv')
    test_file = os.path.join(dataset_path, 'testSet.tsv')
    if dataset_str == 'am':
        label_header = 'label_cateogory'
        nodes_header = 'proxy'
    elif dataset_str == 'aifb':
        label_header = 'label_affiliation'
        nodes_header = 'person'
    elif dataset_str == 'mutag':
        label_header = 'label_mutagenic'
        nodes_header = 'bond'
    elif dataset_str == 'bgs':
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'
    else:
        raise NameError('Dataset name not recognized: ' + dataset_str)

    edge_file = os.path.join(dataset_path, 'edges.npz')
    labels_file = os.path.join(dataset_path, 'labels.npz')
    train_idx_file = os.path.join(dataset_path, 'train_idx.npy')
    test_idx_file = os.path.join(dataset_path, 'test_idx.npy')
    # train_names_file = os.path.join(dataset_path, 'train_names.npy')
    # test_names_file = os.path.join(dataset_path, 'test_names.npy')
    # rel_dict_file = os.path.join(dataset_path, 'rel_dict.pkl')
    # nodes_file = os.path.join(dataset_path, 'nodes.pkl')

    if os.path.isfile(edge_file) and os.path.isfile(labels_file) and \
            os.path.isfile(train_idx_file) and os.path.isfile(test_idx_file):

        # load precomputed adjacency matrix and labels
        all_edges = np.load(edge_file)
        num_node = all_edges['n'].item()
        edge_list = all_edges['edges']
        num_rel = all_edges['nrel'].item()

        print('\t\t - Number of nodes: ', num_node)
        print('\t\t - Number of edges: ', len(edge_list))
        print('\t\t - Number of relations: ', num_rel)

        labels = _load_sparse_csr(labels_file)
        labeled_nodes_idx = list(labels.nonzero()[0])

        print('\t\t - Number of classes: ', labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)

        # train_names = np.load(train_names_file)
        # test_names = np.load(test_names_file)
        # relations_dict = pkl.load(open(rel_dict_file, 'rb'))

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep='\t', encoding='utf-8')
        labels_train_df = pd.read_csv(train_file, sep='\t', encoding='utf8')
        labels_test_df = pd.read_csv(test_file, sep='\t', encoding='utf8')

        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            nodes = list(subjects.union(objects))
            num_node = len(nodes)
            num_rel = len(relations)
            num_rel = 2 * num_rel + 1 # +1 is for self-relation

            assert num_node < np.iinfo(np.int32).max
            print('Number of nodes: ', num_node)
            print('Number of relations: ', num_rel)

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            edge_list = []
            # self relation
            for i in range(num_node):
                edge_list.append((i, i, 0))

            for i, (s, p, o) in enumerate(reader.triples()):
                src = nodes_dict[s]
                dst = nodes_dict[o]
                assert src < num_node and dst < num_node
                rel = relations_dict[p]
                edge_list.append((src, dst, 2 * rel))
                edge_list.append((dst, src, 2 * rel + 1))

            # sort indices by destination
            edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
            edge_list = np.array(edge_list, dtype=np.int)
            print('Number of edges: ', len(edge_list))

            np.savez(edge_file, edges=edge_list, n=np.array(num_node), nrel=np.array(num_rel))

        nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                        nodes_dict.items()}

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        print('{} classes: {}'.format(len(labels_set), labels_set))

        labels = sp.lil_matrix((num_node, len(labels_set)))
        labeled_nodes_idx = []

        print('Loading training set')

        train_idx = []
        train_names = []
        for nod, lab in zip(labels_train_df[nodes_header].values,
                            labels_train_df[label_header].values):
            nod = np.unicode(to_unicode(nod))  # type: unicode
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                train_idx.append(nodes_u_dict[nod])
                train_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        print('Loading test set')

        test_idx = []
        test_names = []
        for nod, lab in zip(labels_test_df[nodes_header].values,
                            labels_test_df[label_header].values):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                test_idx.append(nodes_u_dict[nod])
                test_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        labeled_nodes_idx = sorted(labeled_nodes_idx)
        labels = labels.tocsr()
        print('Number of classes: ', labels.shape[1])

        _save_sparse_csr(labels_file, labels)

        np.save(train_idx_file, train_idx)
        np.save(test_idx_file, test_idx)

        # np.save(train_names_file, train_names)
        # np.save(test_names_file, test_names)

        # pkl.dump(relations_dict, open(rel_dict_file, 'wb'))

    # end if
    print("\t>> Done with loading and preprcessing data??")
    print("\t>> Tensors: ")

    for x in ["num_node", "edge_list", "num_rel", "labels", "labeled_nodes_idx", "train_idx", "test_idx"]:
        print("\t\t>> ", x)
    return num_node, edge_list, num_rel, labels, labeled_nodes_idx, train_idx, test_idx


if __name__ == "__main__":
    data=_load_data(dataset_str="aifb", dataset_path="/home/msabrishami/.dgl/aifb")
    name = "aifb"
    dataset_path = "/home/msabrishami/.dgl/aifb"
    num_nodes, edges, num_rels, labels, labeled_nodes_idx, train_idx, test_idx = _load_data(name, dataset_path)
'''
graph_file = os.path.join(dataset_path, '{}_stripped.nt.gz'.format(dataset_str))
task_file = os.path.join(dataset_path, 'completeDataset.tsv')
train_file = os.path.join(dataset_path, 'trainingSet.tsv')
test_file = os.path.join(dataset_path, 'testSet.tsv')
edge_file = os.path.join(dataset_path, 'edges.npz')
labels_file = os.path.join(dataset_path, 'labels.npz')
train_idx_file = os.path.join(dataset_path, 'train_idx.npy')
test_idx_file = os.path.join(dataset_path, 'test_idx.npy')


all_edges = np.load(edge_file)
num_node = all_edges['n'].item()
edge_list = all_edges['edges']
num_rel = all_edges['nrel'].item()

all_edges = np.load(edge_file)
num_node = all_edges['n'].item()
edge_list = all_edges['edges']
num_rel = all_edges['nrel'].item()

print('Number of nodes: ', num_node)
print('Number of edges: ', len(edge_list))
print('Number of relations: ', num_rel)

# labels = _load_sparse_csr(labels_file)
loader = np.load(label_files)
labels = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'], dtype=np.float32)

labeled_nodes_idx = list(labels.nonzero()[0])

print('Number of classes: ', labels.shape[1])

train_idx = np.load(train_idx_file)
test_idx = np.load(test_idx_file)


if bfs_level > 0:
    print("removing nodes that are more than {} hops away".format(bfs_level))
    row, col, edge_type = edges.transpose()
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
    bfs_generator = _bfs_relational(A, labeled_nodes_idx)
    lvls = list()
    lvls.append(set(labeled_nodes_idx))
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
    self.src, self.dst, self.edge_type = edges.transpose()
 
# return num_node, edge_list, num_rel, labels, labeled_nodes_idx, train_idx, test_idx
# This is all what we wanted: 
# self.num_nodes = num_nodes
# edges = edge_list
# self.num_rels = num_rels
# self.labels = labels
# labeled_nodes_idx = labeled_nodes_idx
# self.train_idx = train_idx
# self.test_idx = test_idx

'''

