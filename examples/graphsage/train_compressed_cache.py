import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
from model import SAGE
from utils.load_graph import *
import os
from torch.optim.lr_scheduler import ExponentialLR
from utils.compresser import Compresser
import utils.storage as storage

os.environ["OMP_NUM_THREADS"] = str(16)
th.multiprocessing.set_sharing_strategy('file_system')
class intsampler(dgl.dataloading.MultiLayerNeighborSampler):
    def __init__(self, fanouts, dev_id, replace=False, return_eids=False):
        super().__init__(fanouts, replace, return_eids)  
        self.dev_id = dev_id

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # input_nodes, output_nodes, blocks
        blocks = super().sample_blocks(g, seed_nodes, exclude_eids)
        blocks = [block.int() for block in blocks]
        return blocks


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device, compresser, cacher):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()

    with th.no_grad():
        perm = th.randperm(len(val_nid))
        val_nid = val_nid[perm][:len(val_nid)//3]
        pred = th.zeros(g.num_nodes(), model.n_classes)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            val_nid,
            sampler,
            use_ddp=False,
            device=device,
            # device=None,
            batch_size=50,
            shuffle=False,
            drop_last=False,
            num_workers=0)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            # print(blocks)
            blocks = [block.to(device) for block in blocks]

            batch_inputs, batch_labels, _ = load_subtensor(nfeat, labels,
                                                        output_nodes, input_nodes, device, compresser, cacher)                                                            
            result = model(blocks, batch_inputs).cpu()
            pred[output_nodes] = result
    model.train()

    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id, compresser, cacher):
    """
    Extracts features and labels for a subset of nodes.
    """
    t0 = time.time() 
    # exp = th.index_select(nfeat, 0, input_nodes.to(nfeat.device))
    exp = cacher.fetch_data(input_nodes)    
    t1 = time.time()
    exp = exp.to(dev_id, non_blocking=True)
    t2 = time.time()        

 
    batch_inputs = compresser.decompress(exp, dev_id)
    batch_labels = th.index_select(labels, 0, seeds.to(labels.device))    
    batch_labels = batch_labels.to(dev_id, non_blocking=True)      
    t3 = time.time()

    return batch_inputs, batch_labels, [t1-t0, t2-t1, t3-t2]

#### Entry point

def run(proc_id, n_gpus, args, devices, data, compresser):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    n_classes, train_g, val_g, test_g = data

    if args.inductive:
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels').long()
        val_labels = val_g.ndata.pop('labels').long()
        test_labels = test_g.ndata.pop('labels').long()
    else:
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels').long()        

    if args.data_gpu:
        train_nfeat = train_nfeat.to(dev_id)
        train_labels = train_labels.to(dev_id)


    train_mask = train_g.ndata.pop('train_mask')
    val_mask = val_g.ndata.pop('val_mask')
    test_mask = test_g.ndata.pop('test_mask')
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    print(len(train_nid), len(val_nid), len(test_nid))

    # Create Pyth DataLoader for constructing blocks
    sampler = intsampler(
        [int(fanout) for fanout in args.fan_out.split(',')], dev_id)    

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        use_ddp=n_gpus > 1,
        device=dev_id if args.num_workers == 0 and args.sample_gpu else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
        )


    t2fid = None
    embed_names = ['features']    
    cacher = storage.GraphCacheServer(train_nfeat, train_g.num_nodes(), t2fid,  dev_id)
    del train_nfeat
    train_nfeat = None
    # cacher.log = True

    # Define model and optimizer
    model = SAGE(compresser.feat_dim, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, MLP=(args.dataset=="mag240m"), bn=False)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    if proc_id == 0:
        with open("results/loss_acc.txt", "a+") as f:
            print("============\nGraphSAGE", args.mode, args.length, args.width, args.dataset, args.fan_out, sep="\t", file=f)
            print(args, file=f)
    start_time = time.time()
    best_eval = 0
    best_test = 0
    avg = 0
    iter_tput = []
    time_log = []
    acc_mean = 0
    loss_mean = 0
    tot_steps = len(dataloader)
    for epoch in range(args.num_epochs):
        if n_gpus > 1:
            dataloader.set_epoch(epoch)
        model.train()    
        tic = time.time()
        t4 = time.time()
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

            t0 = time.time()
            # Load the input features as well as output labels
            batch_inputs, batch_labels, tlist = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, dev_id, compresser, cacher)
            if proc_id == 0 and step==0 and epoch==0:
                print(blocks)
            t1 = time.time()
            blocks = [block.int().to(dev_id, non_blocking=True) for block in blocks]
            t2 = time.time()
            # Compute loss and prediction
            # print(batch_inputs.device, batch_labels.device)
            batch_pred = model(blocks, batch_inputs)
            t5 = time.time()
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t3 = time.time()        
            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if acc_mean == None:
                acc_mean = float(compute_acc(batch_pred, batch_labels))
                loss_mean = float(loss.item())
            if step % 10 == 0:
                acc_mean = acc_mean*0.9+float(compute_acc(batch_pred, batch_labels))*0.1
                loss_mean = loss_mean*0.9+float(loss.item())*0.1
            
            if step % args.log_every == 0 and proc_id == 0:
                acc_mean = acc_mean*0.95+float(compute_acc(batch_pred, batch_labels))*0.05
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss_mean, acc_mean, np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
            if epoch == 0 and step == 1:
                cacher.auto_cache(g, embed_names, args.cache_rate, train_nid) 
            del batch_inputs

            if epoch>0 and step>5 and step != tot_steps-1:
                time_list = [time.time()-t4, t0-t4, t1-t0, t2-t1, t5-t2, time.time()-t5, -1]
                time_list.extend(tlist)
                time_log.append(time_list)       
            t4 = time.time()
            if proc_id == 0:
                tic_step = time.time()
            # print(step)
            # if n_gpus > 1:
            #     th.distributed.barrier()
            
        # if n_gpus > 1:
        #     th.distributed.barrier()

        toc = time.time()
        if proc_id == 0:            
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 2:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:   
                eval_acc = 0  
                test_acc = 0             
                if n_gpus == 1:

                    eval_acc = evaluate(
                        model, val_g, train_nfeat, val_labels, val_nid, devices[0], compresser, cacher)
                    test_acc = evaluate(
                        model, test_g, train_nfeat, test_labels, test_nid, devices[0], compresser, cacher)
                else:

                    eval_acc = evaluate(
                        model.module, val_g, train_nfeat, val_labels, val_nid, devices[0], compresser, cacher)
                    if args.dataset!="mag240m":
                        test_acc = evaluate(
                            model.module, test_g, train_nfeat, test_labels, test_nid, devices[0], compresser, cacher)                                                         

                print('\nEval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))
                with open("results/loss_acc.txt", "a+") as f:
                    print(epoch, "{:.5f}\t{:.5f}\t{:.5f}".format(loss_mean, eval_acc, test_acc), sep="\t", file=f)                  
                if test_acc>best_test:
                    best_test = test_acc

                if eval_acc>best_eval:
                    best_eval = eval_acc                    
        # print("GPU", dev_id, "hit rate", 1-cacher.get_miss_rate())

        scheduler.step()
    # if n_gpus > 1:
    #     th.distributed.barrier()

    print('\nTraining Time(s): {:.4f}'.format(time.time() - start_time))

    if proc_id == 0:    
        print('Avg epoch time: {}'.format(avg / (epoch - 1)))
        with open("results/time_log.txt", "a+") as f:
            for i in np.mean(time_log[3:], axis=0):
                print("{:.5f}".format(i), sep="\t", end="\t", file=f)
            print(avg / (epoch - 1), args.mode, args.length, args.width, args.dataset, args.num_workers, args.gpu, args.batch_size, "GraphSAGE_cache", sep="\t", file=f)         
    if proc_id == 0:
        with open("results/acc.txt", "a+") as f:
            print("GraphSAGE", args.mode, args.length, args.width, args.dataset, args.fan_out,  float(best_eval), float(best_test), sep="\t", file=f)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='1',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=51)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,10,15')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=10)    
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=8,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")                         
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Inductive learning setting")                              
    argparser.add_argument('--data-gpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--cache-rate', type=float, default=1.0)                            
    argparser.add_argument('--mode', type=str, default='sq')
    argparser.add_argument('--width', type=int, default=1)
    argparser.add_argument('--length', type=int, default=1)                                
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()

    elif args.dataset == 'citeseer':
        g, n_classes = load_citeseer()

    elif args.dataset == 'cora':
        g, n_classes = load_cora()

    elif args.dataset == 'amazon':
        g, n_classes = load_amazon()  

    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products', root="/data/graphData/original_dataset")
    # elif args.dataset == 'ogbn-papers100m':
    #     g, n_classes = load_ogbn_papers100m_in_subgraph()       
    # elif args.dataset == 'mag240m':
    #     g, n_classes = load_mag240m_in_subgraph()            
    elif args.dataset == 'ogbn-papers100m':
        g, n_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)         
    elif args.dataset == 'mag240m':
        g, n_classes, feats = load_mag240m()  
        g.ndata["test_mask"] = g.ndata["val_mask"]         
    else:
        raise Exception('unknown dataset')

    print(g, n_classes)

####################################################################################################################
    compresser = Compresser(args.mode, args.length, args.width)
    if args.dataset=="mag240m":
        
        g.ndata["features"] = compresser.compress(feats, args.dataset, batch_size=50000)
    else:
        g.ndata["features"] = compresser.compress(g.ndata.pop("features"), args.dataset)
    print(g.ndata['features'][:3])

#     elif args.dataset == 'ogbn-products':
#         g, n_classes = load_ogb('ogbn-products', root="/data/graphData/original_dataset")
#     elif args.dataset == 'ogbn-papers100m':
#         g, n_classes = load_ogbn_papers100m_in_subgraph()       
#     elif args.dataset == 'mag240m':
#         g, n_classes = load_mag240m_in_subgraph()     
#     # elif args.dataset == 'ogbn-papers100m':
#     #     g, n_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
#     #     srcs, dsts = g.all_edges()
#     #     g.add_edges(dsts, srcs)         
#     # elif args.dataset == 'mag240m':
#     #     g, n_classes, feats = load_mag240m()                    
#     else:
#         raise Exception('unknown dataset')


#     print(g, n_classes)
# ####################################################################################################################
#     compresser = Compresser(args.mode, args.length, args.width)
#     # if args.dataset=="mag240m":
        
#     #     g.ndata["features"] = compresser.compress(feats, args.dataset, batch_size=50000)
#     # else:
#     g.ndata["features"] = compresser.compress(g.ndata.pop("features"), args.dataset)

####################################################################################################################

    # train_nid = g.ndata['train_mask'].nonzero().squeeze()
    # from dgl import function as fn
    # # reversed_g = dgl.reverse(g, copy_ndata=False)
    # print(g)
    # probability = th.zeros(g.num_nodes())
    # weight = 1.0
    # probability[train_nid] = weight
    # # print(th.min(g.out_degrees()))
    # # print(g.out_degrees(th.arange(100)))
    # print(g.out_degrees(train_nid))
    # # print(g.in_degrees(th.arange(100)))
    # # print(g.out_degrees(th.arange(100)))
    # # print(g.in_degrees(th.arange(100)))
    # affected_nodes = train_nid
    # # g.ndata["_d"] = g.out_degrees().to(th.float32)

    # fan_out = [int(fanout) for fanout in args.fan_out.split(',')]
    # for l in range(args.num_layers):
    #     # weight *= 0.1
    #     print("layer", l)

    #     pp = probability.mul(th.minimum(th.tensor(fan_out[args.num_layers-l-1]).div(g.out_degrees().to(th.float32)), th.ones(g.num_nodes())))
    #     print(pp.sum())
    #     src, dst = g.in_edges(affected_nodes)
    #     bs = 1000000
    #     for i in range(0, len(src), bs):
    #         probability[src[i:i+bs]] += pp[dst[i:i+bs]]
    #     affected_nodes = src.unique()

    #     # g.ndata["_P"] = g.ndata["_P"] + g.ndata["_p"]
    # g.ndata["_P"] = probability
    # # del g
    # print(g.ndata["_P"].sum())
    # print(g.ndata["_P"][10:300])
    # # g.ndata["_P"]
    # # g.ndata.pop("_p")
    # # g.ndata.pop("_tp")

    # print(g)


    accs = []

    g = g.formats("csc")  
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    # train_
    # val_g.create_formats_()
    # test_g.create_formats_()
    # Pack data
    data = n_classes, train_g, val_g, test_g      

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data, compresser)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data, compresser))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()




