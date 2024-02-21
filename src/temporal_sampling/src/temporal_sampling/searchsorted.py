#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

# trying to import the CPU searchsorted
SEARCHSORTED_CPU_AVAILABLE = False

# trying to import the CUDA searchsorted
SEARCHSORTED_GPU_AVAILABLE = True
try:
    from temporal_sampling.src.temporal_sampling.cuda import searchsorted_cuda_wrapper
except ImportError:
    SEARCHSORTED_GPU_AVAILABLE = False


def sample_with_pad(frontier_nid: torch.Tensor, frontier_time: torch.Tensor,
                    indptr: torch.Tensor, indices: torch.Tensor, eid: torch.Tensor, timestamp: torch.Tensor,
                    num_sample: int, type_sample: str = 'uniform',
                    dummy_nid: int = None, dummy_eid: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert frontier_nid.shape[0] == frontier_time.shape[0]

    assert type_sample in ['uniform', 'recent']
    if type_sample == 'uniform':
        type_sample = 'u'
    elif type_sample == 'recent':
        type_sample = 'r'

    if dummy_nid is None:
        dummy_nid = indptr.shape[0] - 1
    if dummy_eid is None:
        dummy_eid = eid.max().item() + 1

    result_shape = (frontier_nid.shape[0], num_sample)
    res_node = torch.empty(result_shape, device=frontier_nid.device, dtype=torch.long)
    res_edge = torch.empty(result_shape, device=frontier_nid.device, dtype=torch.long)
    res_time = torch.empty(result_shape, device=frontier_nid.device, dtype=timestamp.dtype)

    if frontier_nid.is_cuda and not SEARCHSORTED_GPU_AVAILABLE:
        raise Exception('temporal_sampling on CUDA device is asked, but it seems '
                        'that it is not available. Please install it')
    if not frontier_nid.is_cuda and not SEARCHSORTED_CPU_AVAILABLE:
        raise Exception('temporal_sampling on CPU is not available. '
                        'Please install it.')

    # breakpoint()
    if frontier_nid.is_cuda:
        searchsorted_cuda_wrapper(frontier_nid, frontier_time,
                                  indptr, indices, eid, timestamp,
                                  res_node, res_edge, res_time,
                                  type_sample, dummy_nid, dummy_eid)
    else:
        raise NotImplementedError

    return res_node, res_edge, res_time
