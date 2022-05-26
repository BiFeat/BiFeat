import torch as th
import math
import numpy as np
import tqdm

from .kmeans import kmeans, get_centers, kmeans_predict
from .packbits import packbits, unpackbits
class Compresser(object):
    def __init__(self, mode="sq", length=1, width=1, device="cpu"):
        self.mode = mode
        self.length = length
        self.width = width
        # self.batch_size = batch_size
        self.device = device
        self.feat_dim = -1
        self.quantized = False
        self.codebooks = None
        self.info = None


    
    def compress(self, features, dn=None, batch_size=-1):
        self.batch_size = batch_size
        if dn:
            self.fn = "/data/giant_graph/quantized/" + dn + "_" + self.mode + "_" + str(self.length) + "_" + str(self.width) + ".pkl"
        import os 
        if dn and os.path.exists(self.fn):
            (compressed, self.codebooks, self.quantized, self.info, self.feat_dim) = th.load(self.fn)
            return compressed
        else:

            if self.mode == "sq":
                compressed = self.sq_compresser(features)

            elif self.mode == "vq":
                if self.batch_size == -1:
                    compressed = self.vq_compresser(features)
                else:
                    compressed = self.vq_compresser_batch(features)
            else:
                raise ValueError("mode must be sq or vq")   
            if self.quantized:  
                th.save((compressed, self.codebooks, self.quantized, self.info, self.feat_dim), self.fn)
            return compressed
   
    def vq_compresser(self, features):
        # vector quantization
        self.quantized = True
        self.feat_dim = features.shape[1]
        print("in total ", math.ceil(features.shape[1]/self.width), " parts")
        self.codebooks = th.empty((math.ceil(features.shape[1]/self.width), self.length, self.width))
        if self.length <=256:
            dtype = th.uint8
        elif self.length <=32768:
            dtype = th.int16
        else:
            dtype = th.int32
        cluster_ids = th.empty((features.shape[0], math.ceil(features.shape[1]/self.width)), dtype=dtype)
        

        for i in range(math.ceil(features.shape[1]/self.width)):  
            print("quantizing part ", i)
            X = features[:, i*self.width:i*self.width+self.width]
            dist = X.norm(dim=1, p=2)
            method = "cosine"
            out_num = self.length - 1

            rim = th.quantile(dist[:50000], 0.3/self.length)

            cluster_ids_x = th.empty(X.shape[0], dtype=th.int32)

            inner = th.lt(dist, rim)   
            cluster_ids_x[inner] = self.length-1
            out = th.ge(dist, rim)

            cluster_ids_o, cluster_centers_o = kmeans(
                X=X[out], num_clusters=out_num, distance=method, 
                tol=3e-2*out_num, device=self.device) 
            cluster_ids_x[out] = cluster_ids_o.to(th.int32)
            
            self.codebooks[i, :, :features.shape[1] - i*self.width] = th.cat((cluster_centers_o, th.ones((1, cluster_centers_o.shape[1])).mul_(1e-4)))

            cluster_ids[:, i] = cluster_ids_x
        return cluster_ids


    def vq_compresser_batch(self, features):
        # vector quantization
        batch_size = self.batch_size
        self.quantized = True
        self.feat_dim = features.shape[1]
        

            # codebooks = vq_batch.gen_codebook(th.from_numpy(dataset.paper_feat[:65536]).float(), width, lens, device)
            # graph.ndata["features"] = th.empty((dataset.num_papers, math.ceil(dataset.num_paper_features/width)), dtype=dtype)
        
        print("in total ", math.ceil(features.shape[1]/self.width), " parts")
        self.codebooks = th.empty((math.ceil(features.shape[1]/self.width), self.length, self.width))
        if self.length <=256:
            dtype = th.uint8
        elif self.length <=32768:
            dtype = th.int16
        else:
            dtype = th.int32
        
        perm = th.randperm(features.shape[0])

        for i in range(math.ceil(features.shape[1]/self.width)):  
            print("quantizing part ", i)
            X = th.tensor(features[perm[:300000], i*self.width:i*self.width+self.width], dtype=th.float32)
            dist = X.norm(dim=1, p=2)
            method = "cosine"
            out_num = self.length

            rim = th.quantile(dist[:50000], 0.3/self.length)

            cluster_ids_x = th.empty(X.shape[0], dtype=th.int32)

            out = th.ge(dist, rim)

            cluster_centers_o = get_centers(
                X=X[out], num_clusters=out_num, distance=method, 
                tol=3e-2*out_num, device=self.device) 
            
            self.codebooks[i, :, :features.shape[1] - i*self.width] = cluster_centers_o
        del X
        cluster_ids = th.empty((features.shape[0], math.ceil(features.shape[1]/self.width)), dtype=dtype)

        for j in tqdm.trange(math.ceil(features.shape[0]/ batch_size), mininterval=1):
            start = j*batch_size
            end = (j+1)*batch_size
            
            features_ = th.tensor(features[start:end, :], dtype=th.float32)
            for i in range(math.ceil(features.shape[1]/self.width)):  
                method = "cosine"
                X = features_[:, i*self.width:i*self.width+self.width]
                
                cluster_ids_x = kmeans_predict(X, self.codebooks[i], method, device=self.device)

                cluster_ids[start:end, i] = cluster_ids_x
        del features_
        del features
        return cluster_ids



    def sq_compresser(self, features):
        # scalar quantization
        self.feat_dim = features.shape[1]
        # print(features[0:10])

        if not th.is_tensor(features):
            features = th.tensor(features, dtype=th.float16)
        if self.length==32 or (self.length==16 and features.dtype==th.float16):
            self.quantized = False

            return features
        else:
            self.quantized = True


        emin = 0
        emax = 0
        drange = 2**(self.length-1)
        
        if self.length<=8:
            dtype = th.int8
        elif self.length<=16:
            dtype = th.int16
        else:
            dtype = th.int32

        if self.length<8:
            tfeat_dim = int(math.ceil(self.feat_dim/8*self.length))
        else:
            tfeat_dim = self.feat_dim

        t_features = th.empty((features.shape[0], tfeat_dim), dtype=dtype)    
        epsilon = 1e-5
        print("start compressing, precision=", self.length)
        perm = th.randperm(features.shape[0])
        sample = features[perm[:100000]]
        fmin = max(np.percentile(np.abs(sample), 0.5), epsilon)
        fmax = max(np.percentile(np.abs(sample), 99.5), 2*epsilon)
        print(fmin, fmax)

        fmin = th.tensor(fmin)
        fmax = th.tensor(fmax)
        quantize_batch_size = 1000000
        for start in tqdm.trange(0, features.shape[0], quantize_batch_size):
            end = min(features.shape[0], start + quantize_batch_size)
            features_ = features[start:end].to(th.float32)
            
            sign = th.sign(features_)  
            if drange==1:
                features_ = th.where(sign<=0, 0, 1)
            else:
                features_ = th.abs(features_)  
                features_ = th.clip(features_, fmin, fmax)
                exp = th.log2(features_)
                emin = th.log2(fmin)
                emax = th.log2(fmax).add(epsilon)                      
     
                exp = th.floor((exp - emin)/(emax-emin)*drange)
                if self.length<8:
                    features_ = th.where(sign<=0, drange-1-exp, exp+drange)
                else:
                    features_ = th.where(sign<=0, -1-exp, exp)

            if self.length<8:
                t_features[start:end] = packbits(features_.to(th.uint8), mask=(1 << self.length) - 1)
            elif self.length==8:
                t_features[start:end] = th.tensor(features_).to(th.int8)
            elif self.length<=16:
                t_features[start:end] = th.tensor(features_).to(th.int16)
            else:
                t_features[start:end] = th.tensor(features_).to(th.int32)
            del features_

        mean = features[:10000].float().norm(1).div(features[:10000].shape[0]*features.shape[1]) 
        if mean < 0.1:
            mean += 0.1
        print(emin,emax, mean)
        info = th.zeros(4)
        info[0] = emin
        info[1] = emax
        info[2] = mean
        info[3] = drange  
        self.info = info
        del features      
        return t_features

    def decompress(self, compressed_features, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device        
        if self.quantized:
            if self.mode == "vq":
                return self.vq_decompresser(compressed_features, device)
            elif self.mode == "sq":
                return self.sq_decompresser(compressed_features, device)
            else:
                raise ValueError("mode should be vq or sq")
        else:
            return compressed_features.to(th.float32).to(device)

    def vq_decompresser(self, compressed_features, device):

        compressed_features = compressed_features.to(device).to(th.int64)
        self.codebooks = self.codebooks.to(device)
        num_parts = self.codebooks.shape[0]
        width = self.width
        decompressed = th.empty((compressed_features.shape[0], self.feat_dim), dtype=th.float32, device=device)
        for i in range(num_parts-1):
            h = i*width
            t = (i+1)*width
            decompressed[:, h:t] = th.index_select(self.codebooks[i], 0, compressed_features[:, i].flatten())
        decompressed[:, (num_parts-1)*width:] = th.index_select(self.codebooks[num_parts-1, :, :self.feat_dim -(num_parts-1)*width], 0, compressed_features[:, num_parts-1].flatten())
        return decompressed

    def sq_decompresser(self, compressed_features, device):

        self.info = self.info.to(device)
        emin = self.info[0]
        emax = self.info[1]
        mean = self.info[2]
        drange = self.info[3]

        exp = compressed_features.to(device)
        if self.length<8:
            exp = unpackbits(exp, mask=2*drange-1, shape=[exp.shape[0], self.feat_dim], dtype=th.uint8)
        if self.length>1:
            if self.length<8:
                exp = exp.to(th.float32) - drange
            exp = exp.add(0.5) 

            sign = th.sign(exp)
            decompressed = th.exp2(exp.abs_().mul_((emax-emin)/drange).add_(emin)).mul_(sign) 
        else:
            decompressed = (exp.to(th.float32).sub_(0.5)).mul_(2*mean)
        return decompressed




if __name__ == "__main__":
    compresser = Compresser("sq", 1, 8)
    features = th.randn(5000, 16)
    compressed_features = compresser.compress(features)
    decompressed_features = compresser.decompress(compressed_features)
    print(features, decompressed_features)
    print(features.shape, compressed_features.shape, decompressed_features.shape)
    print(features.abs().mean(), (decompressed_features- features).abs().mean())



