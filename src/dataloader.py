#  Copyright (c) Meta, Inc. and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import dgl
import torch
import globals
from cache import HistoricalCacheConfig

from temporal_sampling import sample_with_pad
from utils import DenseTemporalBlock


class DataLoader:
    """
        Negative destination node sampling:
            Randomly sample a node. if bipartite graph (wiki, reddit, etc.), sample from the same partition of the positive destination node.
        Training mode:
            Order of root nodes are as specified in the config file
            Provide mini-batches with 1 randomly sampled negative destination nodes
        Val/Test mode:
            Order of root nodes are always 'chorno' (follows time order)
            Provide mini-batches with 49 fixed sampled negative destination nodes
            Fixed means that for any test node, it always have the same 49 negative destination nodes.
            This is to reduce variance in test accuracy.
    """

    def __init__(self, g, fanout,
                 src_nid, dst_nid, timestamp, neg_dst_nid,
                 nfeat, efeat, batch_size,
                 sampler=None,
                 device='cuda', mode='train',
                 eval_neg_dst_nid=None,
                 type_sample='uniform',
                 unique_frontier=False,
                 order='chorno',
                 edge_deg=None,
                 cached_ratio=0.3,
                 enable_cache=False,
                 pure_gpu=False,):
        self.g = [el.to(device) for el in g]  # [indptr, indices, eid, timestamp]
        self.fanout = fanout  # e.g. [10, 5]

        assert src_nid.shape[0] == dst_nid.shape[0] == timestamp.shape[0]
        self.src_nid = src_nid.to(device)
        self.dst_nid = dst_nid.to(device)
        self.timestamp = timestamp.to(device)
        self.neg_dst_nid = neg_dst_nid.to(device)
        self.order = order

        if edge_deg is not None:
            # initialize root edge probability
            if order == 'edge_inv':
                # clipped_node = edge_deg < 3 # remove node with less than 3 neighbors
                root_prob = torch.reciprocal(edge_deg + 10)  # add an offset
                # root_prob[clipped_node] = 0
                root_prob /= root_prob.sum()
                self.root_prob = root_prob.to(device)
            elif order == 'edge_noneinv':
                root_prob = edge_deg.float()
                root_prob /= root_prob.sum()
                self.root_prob = root_prob.to(device)

        if self.order.startswith('gradient'):
            self.init_gradient = True
            if self.order == 'gradient':
                self.gradient_offset = 0.1
            else:
                self.gradient_offset = float(self.order.split('-')[1])
            self.gradient = torch.zeros(src_nid.shape[0], dtype=torch.float, device=device)
            self.root_prob = torch.ones(self.src_nid.shape[0], device=device) / self.src_nid.shape[0]

        self.pure_gpu = pure_gpu  # whole edge feature on device
        self.enable_cache = False if pure_gpu or efeat is None else enable_cache

        self.nfeat = nfeat.to(device) if nfeat is not None else None
        if pure_gpu:
            self.efeat = efeat.to(device) if efeat is not None else None
        else:
            self.efeat = efeat.pin_memory() if efeat is not None else None

        if self.enable_cache:
            num_cached_edges = int(efeat.shape[0]*cached_ratio)
            self.cache = HistoricalCacheConfig(efeat.shape[0], num_cached_edges)
            self.cached_efeat = torch.empty((num_cached_edges, efeat.shape[1]), dtype=efeat.dtype, device=device)
        else:
            self.cached_efeat = None

        self.batch_size = batch_size
        self.device = device

        self.sampler = sampler

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        if mode == 'train':
            self.num_neg_dst = 1
        else:
            self.num_neg_dst = 9 if mode == 'val' else 49
            assert eval_neg_dst_nid is not None
            self.eval_neg_dst_nid = eval_neg_dst_nid.to(device)
        assert type_sample in ['uniform', 'recent']
        self.type_sample = type_sample

        self.unique_frontier = unique_frontier

        self.dummy_nid = self.nfeat.shape[0] - 1 if nfeat is not None else g[0].shape[0] - 1
        self.dummy_eid = self.efeat.shape[0] - 1 if efeat is not None else g[1].max() + 1

        self.edge_idx = None
        self.start = None
        self.end = None
        self.epoch_end = None
        self.reset()

    def _collate(self, batch_idx, log_cache_hit_miss=False):
        # construct root nodes: (pos_src || pos_dst || neg_dst)
        # TODO: Change neg_idx generation. Currently, we randomly sample negative nodes with replacement

        if self.mode == 'train': globals.timer.start_scope_sample()
        if self.mode == 'train':
            neg_idx = torch.randint(low=0, high=self.neg_dst_nid.shape[0], size=(self.num_neg_dst * len(batch_idx),))
            neg_dst_nid = self.neg_dst_nid[neg_idx]
        else:
            neg_dst_nid = self.eval_neg_dst_nid[batch_idx[0] * self.num_neg_dst
                                                : batch_idx[0] * self.num_neg_dst + self.num_neg_dst * len(batch_idx)]
        root_nid = torch.cat([self.src_nid[batch_idx], self.dst_nid[batch_idx], neg_dst_nid])
        root_ts = torch.cat([self.timestamp[batch_idx], self.timestamp[batch_idx],
                             self.timestamp[batch_idx].tile(self.num_neg_dst)])
        blocks = []
        if self.mode == 'train': globals.timer.end_scope_sample()

        """
        1. Get Edge Feature for Each Root Edge 
        1.5 Update block data
        2. Compute Link Encoding for Root Node.
            - Frequency set to one for root node
            - No Identity encoding
        3. Concat with Link Encoding Block for MLPMixer / Apply TGAT's self attention mechanism 
        """

        for i, num_sample in enumerate(self.fanout):
            if self.mode == 'train': globals.timer.start_scope_sample()
            neigh_nid, neigh_eid, neigh_ts = sample_with_pad(
                root_nid, root_ts,
                self.g[0], self.g[1], self.g[2], self.g[3],
                num_sample, self.type_sample,
                self.dummy_nid, self.dummy_eid
            )
            block = DenseTemporalBlock(root_nid, root_ts, neigh_nid, neigh_eid, neigh_ts,
                                       self.dummy_nid, self.dummy_eid)
            if self.mode == 'train': globals.timer.end_scope_sample()

            # slice feature
            if self.mode == 'train': globals.timer.start_slice()
            block.slice_input_node_features(self.nfeat)
            cached_mask, cache_idx = None, None
            if self.enable_cache:
                cached_mask = self.cache.get_cached_mask(neigh_eid)
                cache_idx = self.cache.get_cache_idx(neigh_eid)
                if log_cache_hit_miss:
                    block.cache_hit_count = int(cached_mask.sum())
                    block.cache_miss_count = torch.numel(cached_mask) - block.cache_hit_count
            block.slice_input_edge_features(self.efeat, self.cached_efeat, cached_mask, cache_idx)
            if self.mode == 'train': globals.timer.end_slice()

            if self.enable_cache:
                self.cache.update(neigh_eid)

            if self.sampler is not None:
                if self.mode == 'train': globals.timer.start_neighbor_sample()
                block = self.sampler(block, i == len(self.fanout) - 1)
                if self.mode == 'train': globals.timer.end_neighbor_sample()

            if self.mode == 'train': globals.timer.start_scope_sample()
            # concat root set after frontier set to avoid affecting local_neigh indices
            if i < len(self.fanout) - 1:
                frontier_nid, frontier_ts = block.get_frontier(self.unique_frontier)
                root_nid, root_ts = torch.cat([frontier_nid, root_nid]), torch.cat([frontier_ts, root_ts])

            # set root_size and gradient_idx for output layer
            if i == 0:
                block.set_root_size(len(batch_idx), self.num_neg_dst)
                block.gradient_idx = batch_idx

            blocks.insert(0, block)
            if self.mode == 'train': globals.timer.end_scope_sample()

        return blocks

    def update_gradient(self, idx, gradient):
        self.gradient[idx] = gradient.squeeze()
        return

    def reset(self, log_cache_hit_miss=False):
        globals.timer.start_cache_reset()
        if self.enable_cache:
            if log_cache_hit_miss:
                oracle_cache_hit_rate = self.cache.get_oracle_hit_rate()
            is_update, cached_eid = self.cache.next_epoch()
            if is_update:
                self.cached_efeat = dgl.utils.gather_pinned_tensor_rows(self.efeat, cached_eid)

        if self.order == 'uniform_random':
            self.edge_idx = torch.randperm(self.src_nid.shape[0], device=self.device)
        elif self.order == 'chorno':
            self.edge_idx = torch.arange(self.src_nid.shape[0], device=self.device)
        elif self.order == 'edge_inv' or self.order == 'edge_noneinv':
            self.edge_idx = torch.multinomial(self.root_prob, self.src_nid.shape[0], replacement=True)
        elif self.order.startswith('gradient'):
            if self.init_gradient:
                self.edge_idx = torch.randperm(self.src_nid.shape[0], device=self.device)
                self.init_gradient = False
            else:
                root_prob = self.gradient + self.gradient_offset
                root_prob /= root_prob.sum()
                self.root_prob = root_prob
                self.edge_idx = torch.multinomial(root_prob, self.src_nid.shape[0], replacement=True)
        else:
            raise NotImplementedError

        self.start = 0
        self.end = self.batch_size
        self.epoch_end = False
        globals.timer.end_cache_reset()
        if log_cache_hit_miss:
            return oracle_cache_hit_rate
        else:
            return

    def get_blocks(self, log_cache_hit_miss=False):
        if self.end == self.src_nid.shape[0]:
            self.epoch_end = True
        blocks = self._collate(self.edge_idx[self.start:self.end], log_cache_hit_miss=log_cache_hit_miss)
        self.start = self.end
        self.end = min(self.src_nid.shape[0], self.start + self.batch_size)
        return blocks

