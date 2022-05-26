

# experiments to do

##### Accuracy

test **full precision, SQ, VQ**  results

- node classification: 
  - GraphSAGE(only MAG240M)	**~ 3×3×tune hours**
  - ClusterGCN, FastGCN (Reddit OGBN-Papers100M MAG240M) 
    - **~  2×(0.5+2+3)×3×tune hours**
- link prediction:
  - GraphSAGE(+1 model if possible)
  - 1-2dataset  **~4hours**
- graph regression:
  - GraphSAGE(+1 model if possible)
  - 1-2dataset  **~4hours**





##### Speed up

- convergence(loss&acc VS epoch)  
  - node classification (1~3models ×1~3 datasets)(test acc per epoch) **1~5hours**
- epoch training time
  - GraphSAGE(only MAG240M)
  - ClusterGCN, FastGCN (Reddit OGBN-Papers100M MAG240M) 
  - **full precision, SQ, VQ** (need both SQ and VQ?)
  - **uncached use result of accuracy test, cached need additional ~5hours**
- breakdown, show data loading is bottleneck and the source of speedup 
  - (1~3models ×1~3 datasets) use result of accuracy test
- split graph&cache&quantized (best speed we can get)
  - graphsage × MAG240M **~2hours**



Assume tuning doubles time, expect in total: **around 107hours** , about **54 hours** need high end machine, about half of the time need minor human effort and can be parallelized



## Division

bother **Junyi** try find some link prediction and graph regression example code, only need 1 or 2 small datasets, and try run them with and without feature compression, record the accuracy

**Ping** help check mag240m if possible to run on our machine and later i will update modified fastgcn and clustergcn code, please help run accuracy tests



## Code:

https://github.com/SolarisAdams/GNN_Feature_Quantization/tree/main/graphsage

- graphsage
  - model
  - train_compressed: the train script using compression
  - train_sampling: original train script, for compare
  - utils: folder containing codes for compression 
    - **compresser** : most important, include **compress and decompress** code
    - load_graph ：load and process datasets
    - packbits&kmeans: modules used by compresser
    - process_lsc : script to process mag240m, only need to run once to generate full feature(375GB)

other 2 models(ClusterGCN, FastGCN) needs to be  implemented



#### arguments

**compresser.py**

initializing: 

- mode: "vq" or "sq", selecting vector quantization or scalar quantization
- length: 
  - if mode is sq, length mean the number of bit to use, can be 1,2,4...16,32, if length is 32(or 16 in mag240m dataset), meaning no quantization is done.
  - if mode is vq, length mean the number of codebook entries, normally select big numbers like 1024, 2048, 8192, note that larger the length is, the slower vq would be
- width: for vq mode only, the width of each codebook entries, the features would be split into Ceil(feature_dim / width) parts for vector quantization
- device: the device used for compressing, only work for vq, advise on cpu because gpu isn't much faster, it is also used as default device for dequantization

compress:

- features: the features to be quantized
- dn: dataset name, if set, would cache quantization result
- batch_size: for vq only, read and quantize a batch each time, only mag240m needs , doesn't affect training.

decompress:

- compressed_features: features to be dequantized 
- device: device to perform dequantization, features are loaded into device and dequantize.



train_compressed.py shows the usage of compresser, its additional 3 arguments mode, width, length are used to initialize compresser

```sh
python train_compressed.py --dataset reddit --mode sq --length 32
# training without compression
python train_compressed.py --dataset ogbn-papers100m --mode sq --length 1
# training with SQ, quantized into 1 bit(binary feature)
python train_compressed.py --dataset mag240m --mode vq --width 16 --length 2048
# training with VQ, advise width and length be (16-2048, 16-8192, 12-1024)
# these are practical setup for large scale graphs, however, for Reddit, compress ratio can be higher, like (64-2048, 96-16384)
```



for small tasks only need to test accuracy, we can simply add compresser after data loading

```python
# load graph
g, n_classes = load_graph()            

# process features, no other change needed
compresser = Compresser(args.mode, args.length, args.width)
g.ndata["features"] = compresser.compress(g.ndata.pop("features"))
g.ndata["features"] = compresser.decompress(g.ndata.pop("features"))
```



#### environments:

​	torch 1.8.1

​	dgl 0.7 

​	numpy

​	sklearn



