// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "searchsorted_cuda_kernel.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

template <typename scalar_t>
__device__
int64_t binary_search(int64_t start, int64_t end, scalar_t value, scalar_t *arr) {
    // ensure only sample neighbors earlier than the current timestamp
    // arr = [1, 2, 2, 2, 3, 3, 4, 5, 6], value = 3, index should be 2
    int64_t low = start;
    int64_t high = end;
    value -= 1e-5;

    // Optional, for optimization purpose
    if (value <= arr[low]){
        return low - 1;
    }

    while (low <= high) {
        int64_t mid = low + (high - low) / 2;
        if (value <= arr[mid]) {
            high = mid - 1;
        }
        else if (value > arr[mid]) {
            low = mid + 1;
        }
    }
    return high;
}

__device__
int bitmap_search(int *bitmap, int64_t bitblock_start, int64_t bitmap_index) {
    int bitblock_width = 32;
    int bitblock_index = bitmap_index / bitblock_width;   // find the address of bitmap
    int bit_offset = bitmap_index % bitblock_width;		// position within a address

    int initial_mask = 1;
    int mask = (initial_mask << bit_offset);
    int status = atomicOr(&bitmap[bitblock_index + bitblock_start], mask);

    int is_in = (mask & status) >> bit_offset;  //  checks if the target bit was already set before the atomic operation
    if (is_in != 0) {
        is_in = 1;
    }
    return is_in;
}

template <typename scalar_t>
__global__
void searchsorted_kernel(
        int64_t *frontier_nid, scalar_t *frontier_time,
        int64_t num_sample, char type_sample,
        int64_t *res_node, int64_t *res_edge, scalar_t *res_time,
        int64_t *indptr, int64_t *indices, int64_t *eid, scalar_t *timestamp,
        int64_t dummy_nid, int64_t dummy_eid, curandState *global_state
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int block_size = blockDim.x;  // threads used for one block
    int block_index = blockIdx.x;  // one block per frontier_nid node
    int element_index = threadIdx.x;  // one element per sampled edge

    int res_index = block_size * block_index + element_index;
    int64_t nid = frontier_nid[block_index];
    scalar_t ctime = frontier_time[block_index];

    int64_t earliest = indptr[nid];
    int64_t latest = indptr[nid + 1] - 1;

//     if(blockIdx.x == 0 && threadIdx.x == 0){
//         printf("blockIdx: %d, blockDim: %d, threadIdx: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
//         printf("block_index: %d, block_size: %d, res_index: %d\n", block_index, block_size, res_index);
//         for (int i=0; i<5; i++) {
//             printf("frontier_nid: %ld, frontier_time: %f\n", frontier_nid[i], frontier_time[i]);
//         }
//     }

    if (nid == dummy_nid) {
        res_node[res_index] = dummy_nid;
        res_edge[res_index] = dummy_eid;
        res_time[res_index] = ctime;
        return;
    }

    // do binary search
    __shared__ int64_t pivot;
    if (element_index == 0) {
        pivot = binary_search(earliest, latest, ctime, timestamp);
    }
    __syncthreads();
    auto num_neighbor = pivot - earliest + 1;

//     if(block_index == 0 && element_index == 0){
//         printf("nid: %ld, tid: %d, earliest: %ld, pivot: %ld, latest: %ld\n",
//          nid, tid, earliest, pivot, latest);
//     }

    // sample all
    if (num_neighbor <= num_sample) {
        auto selected = earliest + element_index;
        if (selected <= pivot) {
            res_node[res_index] = indices[selected];
            res_edge[res_index] = eid[selected];
            res_time[res_index] = timestamp[selected];
        }
        else {
            res_node[res_index] = dummy_nid;
            res_edge[res_index] = dummy_eid;
            res_time[res_index] = ctime;
        }
        return;
    }

    // sample one neighbor per thread
    int64_t selected = 0;
    if (type_sample == 'r') {          // most recent neighbor sampling
        selected = pivot - element_index;
    }
    else if (type_sample == 'u') {     // uniform sampling, each thread sample a neighbor
        curandState local_state = global_state[threadIdx.x];
        curand_init(tid, 0, 0, &local_state);

        // init bitmap
        extern __shared__ int bitmap[];
        auto start = element_index;
        auto end = num_neighbor / 32 + 1;

        for (auto i = start; i < end; i += block_size) {
            bitmap[i] = 0;
        }
        __syncthreads();

        // sample neighbor
        int is_in = 1;
        while(is_in == 1) {
            selected = static_cast<int64_t>(curand_uniform(&local_state) * num_neighbor);
            is_in = bitmap_search(bitmap, 0, selected);
        }
        selected += earliest;
    }

//     if(block_index == 0 && element_index == 0){
//         printf("selected: %d\n", selected);
//     }

    res_node[res_index] = indices[selected];
    res_edge[res_index] = eid[selected];
    res_time[res_index] = timestamp[selected];
}


void searchsorted_cuda(
    at::Tensor frontier_nid, at::Tensor frontier_time,
    at::Tensor indptr, at::Tensor indices, at::Tensor eid, at::Tensor timestamp,
    at::Tensor res_node, at::Tensor res_edge, at::Tensor res_time,
    char type_sample, int64_t dummy_nid, int64_t dummy_eid
) {
    auto num_frontier = res_node.size(0);
    auto num_sample = res_node.size(1);

    // 1-d blocks & threads
    auto num_blocks = num_frontier;
    auto num_threads = num_sample;

    // random state, different states between threads inside a block
    curandState *global_state;
    cudaMalloc(&global_state, sizeof(curandState) * num_threads);

    // We clip to 32000, so it needs 1001 integers, which is less than 4096 bytes
    int max_shared_memory = 4096;

//     c10::ScalarType type = frontier_time.scalar_type();
//     std::string type_str = c10::toString(type);
//     printf("type: %s", type_str.c_str());

    AT_DISPATCH_ALL_TYPES(frontier_time.scalar_type(), "searchsorted cuda", ([&] {
        searchsorted_kernel<scalar_t><<<num_blocks, num_threads, max_shared_memory>>>(
            frontier_nid.data_ptr<int64_t>(), frontier_time.data_ptr<scalar_t>(),
            num_sample, type_sample,
            res_node.data_ptr<int64_t>(), res_edge.data_ptr<int64_t>(), res_time.data_ptr<scalar_t>(),
            indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(), eid.data_ptr<int64_t>(), timestamp.data_ptr<scalar_t>(),
            dummy_nid, dummy_eid, global_state
        );
    }));

  }
