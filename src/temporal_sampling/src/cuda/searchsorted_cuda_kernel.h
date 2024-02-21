// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef _SEARCHSORTED_CUDA_KERNEL
#define _SEARCHSORTED_CUDA_KERNEL

#include <torch/extension.h>

void searchsorted_cuda(
    at::Tensor frontier_nid, at::Tensor frontier_time,
    at::Tensor indptr, at::Tensor indices, at::Tensor eid, at::Tensor timestamp,
    at::Tensor res_node, at::Tensor res_edge, at::Tensor res_time,
    char type_sample, int64_t dummy_nid, int64_t dummy_eid
);

#endif
