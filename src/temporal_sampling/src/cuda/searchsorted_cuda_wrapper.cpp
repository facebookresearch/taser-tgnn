// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "searchsorted_cuda_wrapper.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void searchsorted_cuda_wrapper(
    at::Tensor frontier_nid, at::Tensor frontier_time,
    at::Tensor indptr, at::Tensor indices, at::Tensor eid, at::Tensor timestamp,
    at::Tensor res_node, at::Tensor res_edge, at::Tensor res_time,
    char type_sample, int64_t dummy_nid, int64_t dummy_eid
){
    CHECK_INPUT(frontier_nid);
    CHECK_INPUT(frontier_time);
    CHECK_INPUT(indptr);
    CHECK_INPUT(indices);
    CHECK_INPUT(eid);
    CHECK_INPUT(timestamp);
    CHECK_INPUT(res_node);
    CHECK_INPUT(res_edge);
    CHECK_INPUT(res_time);

    searchsorted_cuda(frontier_nid, frontier_time,
                      indptr, indices, eid, timestamp,
                      res_node, res_edge, res_time,
                      type_sample, dummy_nid, dummy_eid);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("searchsorted_cuda_wrapper", &searchsorted_cuda_wrapper, "searchsorted (CUDA)");
}
