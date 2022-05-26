import dgl
import torch as th
import numpy as np
from memory_profiler import profile

def load_reddit():

    # from ogb.nodeproppred import DglNodePropPredDataset

    # print('load', name)
    # data = DglNodePropPredDataset(name="ogbn-products", root="/data/graphData/original_dataset")

    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_pubmed():
    from dgl.data import PubmedGraphDataset

    # load reddit data
    data = PubmedGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_citeseer():
    from dgl.data import CiteseerGraphDataset

    # load reddit data
    data = CiteseerGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_cora():
    from dgl.data import CoraGraphDataset

    # load reddit data
    data = CoraGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      
    return g, data.num_classes

def load_amazon():
    from dgl.data import AmazonCoBuyComputerDataset

    # load reddit data
    data = AmazonCoBuyComputerDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    g.ndata.pop('feat')
    g.ndata.pop('label')      

    train_ratio = 0.2
    val_ratio = 0.2
    test_ratio = 0.6

    N = g.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
    val_mask[val_idx] = True
    test_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask


    return g, data.num_classes

def load_mag240m():
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root = "/data/giant_graph/LSC/")
    num_classes = dataset.num_classes
    graph = dgl.load_graphs("/data/giant_graph/LSC/mag240m_kddcup2021/graph.dgl")[0][0]
    del graph.edata["_ID"]
    del graph.edata["etype"]
    print(graph)
    offset = dataset.num_authors + dataset.num_institutions

    train_idx = dataset.get_idx_split('train')+offset
    val_idx = dataset.get_idx_split('valid')+offset
    test_idx = dataset.get_idx_split('valid')+offset
    # graph = graph.formats("csc") 

    full_feat = np.memmap(
        "/data/giant_graph/LSC/mag240m_kddcup2021/full.pkl", mode='r', dtype='float16',
        shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features))   
    graph.ndata["labels"] = th.zeros((graph.number_of_nodes(),), dtype=th.int64)
    graph.ndata["labels"][offset:] = th.tensor(dataset.all_paper_label, dtype=th.int64)
    # graph.ndata['features'] = full_feat

    # alternatively, you can do the following.

    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_idx] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_idx] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    del dataset
    return graph, num_classes, full_feat

def load_mag240m_subgraph():
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root = "/data/LSC/")
    num_classes = dataset.num_classes
    ei_cites = dataset.edge_index('paper', 'paper')
    graph = dgl.graph((np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]])), num_nodes=121751666)
    print(graph, np.max(ei_cites[1]), np.max(ei_cites[0]))
    graph.ndata["labels"] = th.tensor(dataset.all_paper_label)
    # graph.ndata['features'] = dataset.all_paper_feat


    # alternatively, you can do the following.
    train_idx = dataset.get_idx_split('train')
    val_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('valid')

    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_idx] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_idx] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    del dataset
    return graph, num_classes, dataset.all_paper_feat

def load_ogbn_papers100m_in_subgraph():
    graph = dgl.load_graphs("/data/giant_graph/in_subgraph/ogbn_papers100m.dgl")[0][0]
    return graph, 172

def load_mag240m_in_subgraph():
    graph = dgl.load_graphs("/data/giant_graph/in_subgraph/mag240m.dgl")[0][0]
    return graph, 153

# @profile(precision=4,stream=open('memory_profiler.log','w+'))
def load_ogb(name, root, overload=0, train=-1):
    # from ogb.nodeproppred import DglNodePropPredDataset
    from ogb import nodeproppred
    print('load', name)
    data = nodeproppred.DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]
    # graph = graph.to(th.device("cpu"))
    print(graph.ndata['feat'].shape)
    
    # exit()
    graph.ndata['features'] = graph.ndata['feat']
    
    # print("Gen...")
    if overload>0:
        graph.ndata['features'] = th.zeros(graph.ndata['feat'].shape[0], overload)
    # print("finished")
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    if train>0:
        train_size = int(graph.number_of_nodes()*train)
        perm = th.argsort(graph.ndata['year'].flatten())
        train_nid = perm[:train_size]
        print(train_nid)
    print(len(train_nid), len(val_nid), len(test_nid))
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    # graph.ndata['train_mask'] = ~(graph.ndata['val_mask'] | graph.ndata['test_mask'])
    graph.ndata.pop('feat')
    print('finish constructing', name)
    print(graph.device, num_labels)
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
