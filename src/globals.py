#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import time


class Timer:

    def __init__(self, enable=False, len_cache=1000):
        self.val_i = None
        self.val_time = None
        self.train_i = None
        self.train_time = None
        self.slice_i = None
        self.slice_time = None
        self.neighbor_sample_i = None
        self.neighbor_sample_time = None
        self.scope_sample_i = None
        self.scope_sample_time = None
        self.cache_reset_i = None
        self.cache_reset_time = None
        self.enable = enable
        self.len_cache = len_cache

        if self.enable:
            self.scope_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.scope_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.neighbor_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.neighbor_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.slice_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.slice_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.train_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.train_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.val_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.val_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

            self.cache_reset_s = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]
            self.cache_reset_e = [torch.cuda.Event(enable_timing=True) for _ in range(len_cache)]

        self.reset()

    def reset(self):
        self.scope_sample_time = 0
        self.scope_sample_i = 0

        self.neighbor_sample_time = 0
        self.neighbor_sample_i = 0

        self.slice_time = 0
        self.slice_i = 0

        self.train_time = 0
        self.train_i = 0

        self.val_time = 0
        self.val_i = 0

        self.cache_reset_time = 0
        self.cache_reset_i = 0

    def compute_all(self):
        torch.cuda.synchronize()

        for s, e in zip(self.scope_sample_s[:self.scope_sample_i], self.scope_sample_e[:self.scope_sample_i]):
            self.scope_sample_time += s.elapsed_time(e) / 1000
        self.scope_sample_i = 0

        for s, e in zip(self.neighbor_sample_s[:self.neighbor_sample_i],
                        self.neighbor_sample_e[:self.neighbor_sample_i]):
            self.neighbor_sample_time += s.elapsed_time(e) / 1000
        self.neighbor_sample_i = 0

        for s, e in zip(self.slice_s[:self.slice_i], self.slice_e[:self.slice_i]):
            self.slice_time += s.elapsed_time(e) / 1000
        self.slice_i = 0

        for s, e in zip(self.train_s[:self.train_i], self.train_e[:self.train_i]):
            self.train_time += s.elapsed_time(e) / 1000
        self.train_i = 0

        for s, e in zip(self.val_s[:self.val_i], self.val_e[:self.val_i]):
            self.val_time += s.elapsed_time(e) / 1000
        self.val_i = 0

        for s, e in zip(self.cache_reset_s[:self.cache_reset_i], self.cache_reset_e[:self.cache_reset_i]):
            self.cache_reset_time += s.elapsed_time(e) / 1000
        self.cache_reset_i = 0

    def print(self, prefix):
        self.compute_all()
        print('{}train time:{:.4f}s  val time:{:.4f}s'.format(prefix,
                                                              self.scope_sample_time + self.neighbor_sample_time + self.slice_time + self.train_time,
                                                              self.val_time))
        ans = 'scope sample time:{:.4f}s'.format(self.scope_sample_time)
        if self.neighbor_sample_time > 0:
            ans += '  neighbor sample time:{:.4f}s'.format(self.neighbor_sample_time)
        ans += '  slice time:{:.4f}s'.format(self.slice_time)
        ans += '  prop time:{:.4f}s'.format(self.train_time)
        # if self.cache_reset_time > 0:
        #     ans += '  cache reset time:{:.2f}s'.format(self.cache_reset_time)
        print('{}{}'.format(prefix, ans))

    def set_enable(self):
        if not self.enable:
            self.scope_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.scope_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.neighbor_sample_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.neighbor_sample_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.slice_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.slice_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.train_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.train_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.val_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.val_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.cache_reset_s = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]
            self.cache_reset_e = [torch.cuda.Event(enable_timing=True) for _ in range(self.len_cache)]

            self.enable = True

    def start_scope_sample(self):
        if self.enable:
            self.scope_sample_s[self.scope_sample_i].record()

    def end_scope_sample(self):
        if self.enable:
            self.scope_sample_e[self.scope_sample_i].record()
            self.scope_sample_i += 1
            if self.scope_sample_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.scope_sample_s[:self.scope_sample_i], self.scope_sample_e[:self.scope_sample_i]):
                    self.scope_sample_time += s.elapsed_time(e) / 1000
                self.scope_sample_i = 0

    def start_neighbor_sample(self):
        if self.enable:
            self.neighbor_sample_s[self.neighbor_sample_i].record()

    def end_neighbor_sample(self):
        if self.enable:
            self.neighbor_sample_e[self.neighbor_sample_i].record()
            self.neighbor_sample_i += 1
            if self.neighbor_sample_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.neighbor_sample_s[:self.neighbor_sample_i],
                                self.neighbor_sample_e[:self.neighbor_sample_i]):
                    self.neighbor_sample_time += s.elapsed_time(e) / 1000
                self.neighbor_sample_i = 0

    def start_slice(self):
        if self.enable:
            self.slice_s[self.slice_i].record()

    def end_slice(self):
        if self.enable:
            self.slice_e[self.slice_i].record()
            self.slice_i += 1
            if self.slice_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.slice_s[:self.slice_i], self.slice_e[:self.slice_i]):
                    self.slice_time += s.elapsed_time(e) / 1000
                self.slice_i = 0

    def start_train(self):
        if self.enable:
            self.train_s[self.train_i].record()

    def end_train(self):
        if self.enable:
            self.train_e[self.train_i].record()
            self.train_i += 1
            if self.train_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.train_s[:self.train_i], self.train_e[:self.train_i]):
                    self.train_time += s.elapsed_time(e) / 1000
                self.train_i = 0

    def start_val(self):
        if self.enable:
            self.val_s[self.val_i].record()

    def end_val(self):
        if self.enable:
            self.val_e[self.val_i].record()
            self.val_i += 1
            if self.val_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.val_s[:self.val_i], self.val_e[:self.val_i]):
                    self.val_time += s.elapsed_time(e) / 1000
                self.val_i = 0

    def start_cache_reset(self):
        if self.enable:
            self.cache_reset_s[self.cache_reset_i].record()

    def end_cache_reset(self):
        if self.enable:
            self.cache_reset_e[self.cache_reset_i].record()
            self.cache_reset_i += 1
            if self.cache_reset_i == self.len_cache:
                torch.cuda.synchronize()
                for s, e in zip(self.cache_reset_s[:self.cache_reset_i], self.cache_reset_e[:self.cache_reset_i]):
                    self.cache_reset_time += s.elapsed_time(e) / 1000
                self.cache_reset_i = 0


timer = Timer()
