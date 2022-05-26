import torch
import time


class GraphCacheServer:
    """
    Manage graph features
    Automatically fetch the feature tensor from CPU or GPU
    """

    def __init__(self, nfeats, node_num, nid_map, gpuid):
        """
        Paramters:
                graph:   should be created from `dgl.contrib.graph_store`
                node_num: should be sub graph node num
                nid_map: torch tensor. map from local node id to full graph id.
                                                 used in fetch features from remote
        """
        self.nfeats = nfeats
        self.total_dim = self.nfeats.size(1)
        self.fdtype = self.nfeats[0][0].dtype
        self.gpuid = gpuid
        self.node_num = node_num
        self.nid_map = None
        self.gpu_flag = torch.zeros(self.node_num).bool().cuda(self.gpuid)
        self.gpu_flag.requires_grad_(False)
        self.gidtype = torch.int32
        self.cached_num = 0
        self.capability = node_num

        self.full_cached = False
        self.gpu_fix_cache = None
        with torch.cuda.device(self.gpuid):
            self.localid2cacheid = torch.cuda.LongTensor(node_num).fill_(0)
            self.localid2cacheid.requires_grad_(False)

        # logs
        self.log = False
        self.try_num = 0
        self.miss_num = 0

    def auto_cache(self, dgl_g, embed_names, cache_rate, train_nid):
        """
        Automatically cache the node features
        Params:
                g: DGLGraph for local graphs
                embed_names: field name list, e.g. ['features', 'norm']
        """
        self.gidtype = dgl_g.idtype

        # Step1: get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.gpuid)
        peak_cached_mem = torch.cuda.max_memory_cached(device=self.gpuid)
        total_mem = torch.cuda.get_device_properties(self.gpuid).total_memory
        available = total_mem - peak_allocated_mem - 0.3*peak_cached_mem \
            - 1024 * 1024 * 1024 - self.node_num  # in bytes
        # Stpe2: get capability
        csize = self.nfeats[0][0].element_size()
        self.capability = max(0, int(0.8*available / (self.total_dim * csize)))
        if cache_rate != 1.0:
            self.capability = int(self.node_num*cache_rate)
        # self.capability = 0
        # self.capability = int(6 * 1024 * 1024 * 1024 / (self.total_dim * 4))
        #self.capability = int(self.node_num * 0.8)
        print('Cache Memory: {:.2f}G. Capability: {}'
              .format(available / 1024 / 1024 / 1024, self.capability))
        # Step3: cache
        if self.capability >= self.node_num:
            # fully cache
            print('cache the full graph...')
            full_nids = torch.arange(self.node_num).cuda(self.gpuid)
            self.cache_fix_data(full_nids, self.nfeats, is_full=True)
        else:
            # choose top-cap out-degree nodes to cache
            print('cache the part of graph... caching percentage: {:.4f}'
                  .format(self.capability / self.node_num))

            if "_P" in dgl_g.ndata and True:
                sort_nid = torch.argsort(dgl_g.ndata["_P"], descending=True)
                # dgl_g.ndata.pop("_P")
            else:
                out_degrees = dgl_g.out_degrees()
                sort_nid = torch.argsort(out_degrees, descending=True)

            cache_nid = sort_nid[:self.capability]
            data = self.nfeats[cache_nid]
            self.cache_fix_data(cache_nid, data, is_full=False)

    def cache_fix_data(self, nids, data, is_full=False):
        """
        User should make sure tensor data under every field name should
        have same num (axis 0)
        Params:
                nids: node ids to be cached in local graph.
                                        should be equal to data rows. should be in gpu
                data: dict: {'field name': tensor data}
        """
        rows = nids.size(0)
        self.localid2cacheid[nids] = torch.arange(rows, device=self.gpuid)
        self.cached_num = rows
        self.gpu_fix_cache = data.cuda(self.gpuid)
        # setup flags
        self.gpu_flag[nids] = True
        self.full_cached = is_full

    def fetch_data(self, nids):
        """
        copy feature from local GPU memory or
        remote CPU memory, which depends on feature
        current location.
        --Note: Should be paralleled
        Params:
                nodeflow: DGL nodeflow. all nids in nodeflow should
                                                        under sub-graph space
        """
        if self.full_cached:
            # return self.fetch_from_cache(nids)
            return torch.index_select(self.gpu_fix_cache, 0, nids.to(self.gpuid))

        nids_gpu = nids.to(self.gpuid).to()
        with torch.autograd.profiler.record_function('cache-index'):
            gpu_mask = torch.index_select(self.gpu_flag, 0, nids_gpu)
            nids_in_gpu = nids_gpu[gpu_mask]
            cpu_mask = ~gpu_mask
            nids_in_cpu = torch.masked_select(nids_gpu, cpu_mask).cpu()

        with torch.autograd.profiler.record_function('cache-allocate'):
            with torch.cuda.device(self.gpuid):
                data = torch.empty(
                    (nids_gpu.size(0), self.total_dim), dtype=self.fdtype, device=self.gpuid)

        # for gpu cached tensors: ##NOTE: Make sure it is in-place update!
        with torch.autograd.profiler.record_function('cache-gpu'):
            if nids_in_gpu.size(0) != 0:
                cacheid = torch.index_select(
                    self.localid2cacheid, 0, nids_in_gpu)
                data[gpu_mask] = torch.index_select(
                    self.gpu_fix_cache, 0, cacheid)

        # for cpu cached tensors: ##NOTE: Make sure it is in-place update!
        with torch.autograd.profiler.record_function('cache-cpu'):
            if nids_in_cpu.size(0) != 0:
                data[cpu_mask] = torch.index_select(
                    self.nfeats, 0, nids_in_cpu).to(self.gpuid)

        if self.log:
            self.log_miss_rate(nids_in_cpu.size(0), nids.size(0))
        return data

    def fetch_from_cache(self, nids):
        with torch.autograd.profiler.record_function('cache-gpu'):
            data = torch.index_select(
                self.gpu_fix_cache, 0, nids.to(self.gpuid))
        return data

    def log_miss_rate(self, miss_num, total_num):
        self.try_num += total_num
        self.miss_num += miss_num

    def get_miss_rate(self):
        if self.full_cached:
            return 0.0
        if self.try_num == 0:
            return 0.0
        # print(self.miss_num, self.try_num)
        miss_rate = float(self.miss_num) / self.try_num
        self.miss_num = 0
        self.try_num = 0
        return miss_rate
