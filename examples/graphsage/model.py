import torch as th
import torch.nn as nn
import torch.functional as F
import dgl
import dgl.nn as dglnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm     

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP=False, res=False, bn=False):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP, res, bn)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP, res, bn):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.MLP = MLP        
        self.res = res
        self.bn = bn
        self.layers = nn.ModuleList()   
        if self.bn:
            self.bns = nn.ModuleList()
        method = 'mean'
        # method = 'md'        
        if n_layers > 1:
            in_channel = 2*n_hidden if self.res else n_hidden

            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, method))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(n_hidden))
            for i in range(1, n_layers - 1):
                if self.res and i==1:
                    self.layers.append(dglnn.SAGEConv(n_hidden+in_feats, n_hidden, method))
                else:
                    self.layers.append(dglnn.SAGEConv(in_channel, n_hidden, method))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(n_hidden))
                

            if self.MLP:
                self.layers.append(dglnn.SAGEConv(n_hidden+in_feats if self.res and n_layers==2 else in_channel, n_hidden, method))       
            else:         
                self.layers.append(dglnn.SAGEConv(n_hidden+in_feats if self.res and n_layers==2 else in_channel, n_classes, method))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, method))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if self.MLP:
            self.mlp = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(n_hidden, n_classes),
            )            

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if self.res and l==0:
                h_ = h
            if self.res:
                h_ = h_[:block.number_of_dst_nodes()]
            # print(h.size())
            h = layer(block, h)
            if self.bn and l<self.n_layers-1:
                h = self.bns[l](h)
            # print(h.size())
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
                if self.res:
                    h__ = h_
                    h_ = h
                    h = th.cat((h__, h), dim=1)
        if self.MLP:
            return self.mlp(h)
        else:
            return h

    def inference(self, g, x, device, batch_size, num_workers, prec=32, deq_data=None):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        # if self.res:
        #     out_channel = [2*n_hidden for i in range(self.layers)]
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size*50,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, mininterval=5):
                block = blocks[0]

                block = block.int().to(device)

                if l==0 and prec<32:
                    h = dequantize(x[input_nodes].to(device), prec, deq_data)
                else:
                    h = x[input_nodes].to(th.float32).to(device)          
                # print(l, h.size())        
                h = layer(block, h)
                if self.bn and l<self.n_layers-1:
                    h = self.bns[l](h)                
                # print(h.size())  
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                    
                y[output_nodes] = h.cpu()
            if self.res and l != len(self.layers) - 1:
                if l==0:
                    x = th.cat((dequantize(x.to(device), prec, deq_data).cpu(), y), dim=1)
                else:
                    x = th.cat((y_, y), dim=1)

            else:
                x = y
            # print(x.size())
            # print(y.size())
            y_=y
        if self.MLP:
            y = self.mlp(y)
        return y

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test
