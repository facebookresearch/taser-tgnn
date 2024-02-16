# TASER: Temporal Adaptive Sampling for Fast and Accurate Dynamic Graph Representation Learning

## Setup
1. Setup a Python environment (>=3.11). Install PyTorch (>=2.0.1) and Deep Graph Library (>=1.1).  

2. Install nvcc for cuda compilation. Make sure choose the compatible cuda version with your PyTorch. 
    ```
        conda install cuda -c nvidia/label/cuda-11.8.0
    ```

3. Build temporal_sampling GPU operator
    ```
        cd src/temporal_sampling/
        python setup.py build_ext --inplace
    ```

## Download and Preprocess Dataset
1. Download dataset

2. Convert edge CSV to the [Temporal-CSR](https://arxiv.org/abs/2203.14883) format
    ```
        python src/gen_graph.py --data WIKI
    ```

3. Preprocess negative edges   
    ```
        python src/preprocess.py --data WIKI --clip_root_set
    ```

## TASER+TGNN co-training
```
    python src/train.py --config config_train/tgat_wiki/TGAT.yml \
                        --data WIKI \
                        --gpu 0 \
                        --cache \
                        --cached_ratio 0.2  
```
  Important Arguments: 
  - `--config`: Config of TASER+TGNN reported in the paper. The configs of other datasets/models are under the ```config_train``` folder.    
  - `--data`: The training datasets. Available choices [WIKI, REDDIT, Flight, MovieLens, GDELT]
  - `--cache`: Enable GPU caching
  - `--cached_ratio`: Ratios of node features cached in GPU.

## License
TASER is MIT licensed, as found in the LICENSE file.
