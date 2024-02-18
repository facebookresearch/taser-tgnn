#  Copyright (c) Meta, Inc. and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import globals

class HistoricalCacheConfig:
    """
    Cache config for edge features
    """

    def __init__(self, num_total_edges, num_cached_edges=10000, threshold=0.8):
        self.num_total_edges = num_total_edges
        self.num_cached_edges = num_cached_edges
        self.threshold = threshold
        self.access_counts = None
        self.cached_eid = None
        self.cached_mask = None
        self.cache_idx = None
        self.reset()

    def reset(self, access_count=None, cached_mask=None, cache_idx=None):
        self.access_counts = access_count
        self.cached_mask = cached_mask
        self.cache_idx = cache_idx

        if access_count is None:
            if self.access_counts is None:
                self.access_counts = torch.zeros(self.num_total_edges, dtype=torch.long, device='cuda')
            else:
                self.access_counts.zero_()
        if cached_mask is None:
            if self.cached_mask is None:
                self.cached_mask = torch.zeros(self.num_total_edges, dtype=torch.bool, 
                device='cuda')
            else:
                self.cached_mask.zero_()
        if cache_idx is None:
            if self.cache_idx is None:
                self.cache_idx = torch.empty(self.num_total_edges, dtype=torch.long, device='cuda')
            else:
                self.cache_idx.zero_()

    def get_cached_mask(self, neigh_eid):
        return self.cached_mask[neigh_eid]

    def get_cache_idx(self, neigh_eid):
        return self.cache_idx[neigh_eid]

    def update(self, neigh_eid):
        ids, counts = neigh_eid.unique(return_counts=True)
        self.access_counts[ids] += counts

    def next_epoch(self):
        total_counts = self.access_counts.sum()
        if total_counts == 0:
            cached_eid = torch.randperm(self.num_total_edges)[:self.num_cached_edges].cuda()
        else:
            # prob = self.access_counts / total_counts
            # cached_eid = torch.multinomial(prob, self.num_cached_edges, replacement=False).cuda()
            cached_eid = torch.topk(self.access_counts, k=self.num_cached_edges, sorted=False)[1]

        new_cached_mask = torch.zeros(self.num_total_edges, dtype=torch.bool, device='cuda')
        new_cached_mask[cached_eid] = True
        
        overlap_ratio = torch.logical_and(self.cached_mask, new_cached_mask).sum() / self.num_cached_edges
        print(f'\tCache Overlap Ratio: {overlap_ratio:.3f}')

        if overlap_ratio < self.threshold:
            self.reset(cached_mask=new_cached_mask)
            self.cache_idx[cached_eid] = torch.arange(self.num_cached_edges, device='cuda')
        else:
            # only reset edge access counts
            self.reset(cached_mask=self.cached_mask, cache_idx=self.cache_idx)

        return overlap_ratio < self.threshold, cached_eid

    def get_oracle_hit_rate(self):
        eid = torch.topk(self.access_counts, k=self.num_cached_edges, sorted=False)[1]
        return int(self.access_counts[eid].sum()) / int(self.access_counts.sum())


