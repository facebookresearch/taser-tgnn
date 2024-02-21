#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import time

import dgl
import torch
import numpy as np

import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import probs_to_logits, logits_to_probs, lazy_property

class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real_vector}
    has_enumerate_support = True

    def __init__(self, probs=None, logits=None, validate_args=None):
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        super(Categorical, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if 'probs' in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if 'logits' in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    @property
    def param_shape(self):
        return self._param.size()

    @property
    def mean(self):
        return torch.full(self._extended_shape(), nan, dtype=self.probs.dtype, device=self.probs.device)

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    @property
    def variance(self):
        return torch.full(self._extended_shape(), nan, dtype=self.probs.dtype, device=self.probs.device)

    def sample(self, sample_shape=torch.Size(), replacement=False):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), replacement).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values


def load_data(dataset, root_path='DATA'):
    data_path = os.path.join(root_path, dataset)

    g = np.load(os.path.join(data_path, 'ext_full_clipped.npz'))
    g = [torch.as_tensor(g['indptr']), torch.as_tensor(g['indices']), 
         torch.as_tensor(g['eid']), torch.as_tensor(g['ts'], dtype=torch.float32)]
    
    edge_split = torch.load(os.path.join(data_path, 'edges.pt'))

    nfeat, efeat = None, None
    if os.path.isfile(os.path.join(data_path, 'node_features_pad.pt')):
        nfeat = torch.load(os.path.join(data_path, 'node_features_pad.pt'))
    if os.path.isfile(os.path.join(data_path, 'edge_features_pad.pt')):
        efeat = torch.load(os.path.join(data_path, 'edge_features_pad.pt'))
    return g, edge_split, nfeat, efeat


class DenseTemporalBlock:
    """
    contains mini-batch information for a single TGNN layer
    the nodes with insufficient temporal neighbors are padded with null (all-zero) neighbors
    this class should be stored purely on GPU
    """

    def __init__(self, root_nid, root_ts, neigh_nid, neigh_eid, neigh_ts, dummy_nid, dummy_eid):
        # let this batch has n root nodes, each root node has b temporal neighbors
        self.n = neigh_nid.shape[0]
        self.b = neigh_nid.shape[1]

        # details referred to set_root_size()
        self.src_size = None  # int: number of source nodes
        self.pos_dst_size = None  # int: number of negative pos nodes;
        self.neg_dst_size = None  # int: number of negative destination nodes;
        self.num_neg_dst = None

        assert self.n == root_nid.shape[0] == root_ts.shape[0]
        self.root_nid = root_nid  # torch.long of size (n): node ids of root nodes
        self.root_ts = root_ts  # torch.float of size (n): timestamps of root nodes

        self.neighbor_nid = neigh_nid  # torch.long of size (n, b): temporal neighbor node ids where row i are the temporal neighbors of root_nid[i]
        self.neighbor_eid = neigh_eid  # torch.long of size (n, b): temporal neighbor edge ids where row i are the edges ids of root_id[i]
        self.neighbor_ts = neigh_ts  # torch.float of size (n, b): temporal neighbor timestamps where row i are the timestamps of root_id[i]; note that neighbor_ts[i,;] should be strictly less than root_ts[i]

        self.dummy_nid = dummy_nid
        self.dummy_eid = dummy_eid

        # use unique to reduce frontier set
        # self.frontier_nid = frontier_nid  # torch.long of size (<n*b)
        # self.frontier_ts = frontier_ts  # torch.long of size (<n*b)
        self.local_neigh_idx = None  # torch.long of size (n, b): temporal neighbor indices where row i are the temporal neighbors of root_nid[i], the value of each element correspond to the index of frontier_nid & frontier_ts

        self.root_node_feature = None
        self.neighbor_node_feature = None
        self.neighbor_edge_feature = None

        """
        How padding works:
        For a dataset with V nodes and E edges, we add a dummy node with node id V and a dummy edge with edge id E.
        We pad the node features and edge features matrix with an additional all-zero rows.
        Suppose V=10 and E=20, and we have node 0 at timestamp 2.5 with 3 temporal neighbors but b=5 (we need to pad two neighbors)
            root_nid = [0]
            root_ts = [2.5]
            neighbor_nid = [[10, 10, 1, 6, 4]] (pad with 10)
            neighbor_eid = [[20, 20, 3, 14, 17]] (pad with 20)
            neighbor_ts = [[2.5, 2.5, 1.4, 2.2, 2.3]] (pad with root_ts)
        When we sample neighbors for a padded node in the next layer, it returns all padding hop-2 neighbors (dummy nodes and edges)
        """

        """
        How to sample multiple layers:
        Suppose we have neighbor_nid = [[1, 2 ,3], [5, 6, 7]] for the current layer. In the next layer, we flatten it into a 1-d tensor as the root_nid of the next layer.
        Similar for neighbor_ts.
        """

    @property
    def device(self):
        return self.root_nid.device

    @property
    def is_frontier_unique(self):
        return self.local_neigh_idx is not None

    def get_frontier(self, is_unique=False):
        if is_unique:
            frontier, local_neigh = torch.stack(
                [self.neighbor_nid.flatten().to(torch.float64), self.neighbor_ts.flatten().to(torch.float64)]
            ).unique(dim=1, return_inverse=True)
            self.local_neigh_idx = local_neigh.view(-1, self.b)  # use local_neigh to construct neigh info from frontier
            frontier_nid, frontier_ts = frontier[0].to(self.root_nid.dtype), frontier[1].to(self.root_ts.dtype)
        else:
            frontier_nid, frontier_ts = self.neighbor_nid.flatten(), self.neighbor_ts.flatten()
        return frontier_nid, frontier_ts

    def update(self, action):
        """
        update current block according to the action. The action is the sampled index with size(n, b'), where b' < b.
        """
        assert self.n == action.size(0)
        self.b = action.size(1)
        self.neighbor_nid = torch.gather(self.neighbor_nid, dim=1, index=action)
        self.neighbor_eid = torch.gather(self.neighbor_eid, dim=1, index=action)
        self.neighbor_ts = torch.gather(self.neighbor_ts, dim=1, index=action)
        if self.neighbor_node_feature is not None:
            idx = action.unsqueeze(-1).expand(action.size(0), action.size(1), self.neighbor_node_feature.size(-1))
            self.neighbor_node_feature = torch.gather(self.neighbor_node_feature, dim=1, index=idx)
        if self.neighbor_edge_feature is not None:
            idx = action.unsqueeze(-1).expand(action.size(0), action.size(1), self.neighbor_edge_feature.size(-1))
            self.neighbor_edge_feature = torch.gather(self.neighbor_edge_feature, dim=1, index=idx)

    def slice_input_edge_features(self, efeat, cached_efeat=None, cached_mask=None, cache_idx=None):
        # torch.float of size (n, b, dim_edge_feat), if edge features are not present, then size (n, b, 0)'
        if efeat is not None:
            if cached_efeat is not None and cached_mask is not None and cache_idx is not None:
                self.neighbor_edge_feature = torch.empty((self.n, self.b, efeat.shape[-1]), device='cuda', dtype=efeat.dtype)
                self.neighbor_edge_feature[cached_mask] = cached_efeat[cache_idx[cached_mask]]
                self.neighbor_edge_feature[~cached_mask] = dgl.utils.gather_pinned_tensor_rows(efeat, self.neighbor_eid[~cached_mask])
            else:
                self.neighbor_edge_feature = efeat[self.neighbor_eid.to(efeat.device), :].cuda()
        else:
            self.neighbor_edge_feature = torch.zeros((self.n, self.b, 0), device=torch.device("cuda"))

    def slice_input_node_features(self, nfeat):
        if nfeat is not None:
            self.root_node_feature = nfeat[self.root_nid, :]  # torch.float of size (n, dim_node_feat), if node features are not present, then size (n, 0)
            self.neighbor_node_feature = nfeat[self.neighbor_nid, :]  # torch.float of size (n, b, dim_node_feat), if node features are not present, then size (n, b, 0)
        else:
            self.root_node_feature = torch.zeros((self.root_nid.shape[0], 0), device=torch.device("cuda"))
            self.neighbor_node_feature = torch.zeros((self.neighbor_nid.shape[0], self.neighbor_nid.shape[1], 0), device=torch.device("cuda"))

    def slice_hidden_node_features(self, hidden_feat):
        self.root_node_feature = hidden_feat[-self.n:]  # size (n, dim_node_feat)
        neigh_feat = hidden_feat[:-self.n]
        if self.is_frontier_unique:
            self.neighbor_node_feature = neigh_feat[self.local_neigh_idx, :]  # size (n, b, dim_node_feat)
        else:
            assert neigh_feat.shape[0] == self.n * self.b
            self.neighbor_node_feature = neigh_feat.reshape(self.n, self.b, -1)  # size (n, b, dim_node_feat)

    def set_root_size(self, batch_size, num_neg_per_node):
        """
        Only available for the output block.
        For an output block, root_nid should follow the order of (pos_src || pos_dst || neg_dst);
        the order in neg_dst should be (pos_src_0_neg_dst_0, pos_src_1_neg_dst_0, ..., pos_src_0_neg_dst_1, pos_src_1_neg_dst_1, ...)
        """
        self.src_size = batch_size
        self.pos_dst_size = batch_size
        self.neg_dst_size = batch_size * num_neg_per_node
        self.num_neg_dst = num_neg_per_node
        assert self.n == self.src_size + self.pos_dst_size + self.neg_dst_size


def select_device(choose_idle=True):
    mem_threshold = 20 * 1e9  # 20G
    time_interval = 60  # sec
    while True:
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.mem_get_info(device=i)
            if mem[0] > mem_threshold:
                if not choose_idle or mem[1] - mem[0] < 1 * 1e9:
                    return 'cuda:{}'.format(i)

        print('No GPU available, wait for {} sec'.format(time_interval))
        time.sleep(time_interval)
