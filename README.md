

# Code for BiFeat: Supercharge GNN Training via Graph Feature Quantization


Uses feature quantization to speedup data loading and save memory.

### Files:

- **compresser** : The main component of BiFeat, include **compress and decompress** code
- packbits&kmeans: modules used by compresser

- examples
  - graphsage
    - model
    - train_compressed: the train script using compression
    - train_sampling: original train script, for compare
    - utils: folder containing codes for compression 
      - compresser : include compress and decompress code
      - load_graph ：load and process datasets
      - packbits&kmeans: modules used by compresser
      - process_lsc : script to process mag240m, to generate full feature




### Arguments:

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





