#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch_geometric.nn import inits


class LearnableTimeEncoder(torch.nn.Module):

    def __init__(self, dim):
        super(LearnableTimeEncoder, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(self.dim))

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FixedTimeEncoder(torch.nn.Module):

    def __init__(self, dim):
        super(FixedTimeEncoder, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class FixedFrequencyEncoder(torch.nn.Module):

    def __init__(self, dim, encode_type='sin'):
        super(FixedFrequencyEncoder, self).__init__()

        self.dim = dim
        assert encode_type in ['sin', 'fourier', 'poly']
        self.encode_type = encode_type

    @torch.no_grad()
    def forward(self, freqs):
        device = freqs.device
        if self.encode_type == 'sin':  # sinusoidal_encoding
            div_term = torch.exp(
                torch.arange(0., self.dim, 2, device=device) * -(torch.log(torch.tensor(10000.0)) / self.dim))
            encoded = torch.zeros(freqs.shape[0], self.dim, device=device)
            encoded[:, 0::2] = torch.sin(freqs.unsqueeze(-1) * div_term)
            encoded[:, 1::2] = torch.cos(freqs.unsqueeze(-1) * div_term)
        elif self.encode_type == 'poly':  # polynomial_encoding
            powers = torch.arange(self.dim + 1, device=device).unsqueeze(0)
            encoded = torch.pow(freqs.unsqueeze(-1), powers)
        elif self.encode_type == 'fourier':  # fourier_encoding
            signal = torch.sin(2 * torch.pi * freqs.unsqueeze(-1) * torch.arange(self.dim, device=device))
            spectrum = torch.fft.fft(signal)
            encoded = spectrum.real
        else:
            raise NotImplementedError
        return encoded


class TransformerAggregator(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dim_out,
                 dropout=0.0, time_encoder_type='learnable', att_clamp=10., save_h_neigh_grad=False):
        super(TransformerAggregator, self).__init__()

        self.h_v = None
        self.h_exp_a = None
        self.h_neigh = None

        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.num_head = num_head
        self.dim_out = dim_out

        self.dropout = torch.nn.Dropout(dropout)

        self.att_dropout = torch.nn.Dropout(dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)

        if time_encoder_type == 'learnable':
            self.time_encoder = LearnableTimeEncoder(dim_time)
        elif time_encoder_type == 'fixed':
            self.time_encoder = FixedTimeEncoder(dim_time)
        else:
            raise NotImplementedError

        self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
        self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)

        self.layer_norm = torch.nn.LayerNorm(dim_out)

        self.att_clamp = att_clamp
        self.neigh_grad = save_h_neigh_grad

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_loss(self, log_prob):
        log_prob = log_prob.unsqueeze(2).unsqueeze(3)

        grad_h_neigh = self.h_neigh.grad.detach()
        h_neigh_detach = self.h_neigh.detach()
        h_exp_a_detach = self.h_exp_a.detach()
        h_v_detach = self.h_v.detach()

        coef = (h_exp_a_detach.mean(dim=1)).pow(-3)
        h_neigh = (log_prob * h_exp_a_detach * h_v_detach).mean(dim=1) + \
                  h_neigh_detach * (log_prob * h_exp_a_detach).mean(dim=1)
        h_neigh = coef * h_neigh

        batch_loss = torch.bmm(grad_h_neigh.view(grad_h_neigh.shape[0], 1, -1),
                               h_neigh.view(h_neigh.shape[0], -1, 1))
        # batch_size = int(batch_loss.shape[0] / 6 / 3)
        # idx = torch.cat([torch.arange(batch_size*2*5), torch.arange(batch_size*3*5, batch_size*3*5+batch_size*2)])
        # batch_loss = batch_loss[idx]
        return batch_loss

    def forward(self, block):
        zero_time_feat = self.time_encoder(torch.zeros(block.n, dtype=torch.float32, device=self.device))
        edge_time_feat = self.time_encoder((block.root_ts.unsqueeze(-1) - block.neighbor_ts).flatten())

        # import pdb; pdb.set_trace()
        neighbor_node_feature = block.neighbor_node_feature.view(
            block.neighbor_node_feature.shape[0] * block.neighbor_node_feature.shape[1],
            block.neighbor_node_feature.shape[2]
        )
        neighbor_edge_feature = block.neighbor_edge_feature.view(
            block.neighbor_edge_feature.shape[0] * block.neighbor_edge_feature.shape[1],
            block.neighbor_edge_feature.shape[2]
        )

        h_q = self.w_q(torch.cat([block.root_node_feature, zero_time_feat], dim=1))
        h_k = self.w_k(torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat], dim=1))
        h_v = self.w_v(torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat], dim=1))

        h_q = h_q.view((h_q.shape[0], 1, self.num_head, -1))
        h_k = h_k.view((h_q.shape[0], -1, self.num_head, h_q.shape[-1]))
        h_v = h_v.view((h_q.shape[0], -1, self.num_head, h_q.shape[-1]))

        if self.neigh_grad:
            self.h_v = h_v

        h_att = self.att_act(torch.sum(h_q * h_k, dim=3))

        if self.neigh_grad:
            self.h_exp_a = torch.exp(torch.clamp(h_att, -self.att_clamp, self.att_clamp)).unsqueeze(-1)
            # self.h_exp_a = torch.exp(h_att - h_att.max()).unsqueeze(-1)

        h_att = F.softmax(h_att, dim=1).unsqueeze(-1)
        h_neigh = (h_v * h_att).sum(dim=1)

        if self.neigh_grad:
            self.h_neigh = h_neigh
            if self.training: self.h_neigh.retain_grad()

        h_neigh = h_neigh.view(h_v.shape[0], -1)
        h_out = self.w_out(torch.cat([h_neigh, block.root_node_feature], dim=1))
        h_out = self.layer_norm(torch.nn.functional.relu(self.dropout(h_out)))
        return h_out


class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """

    def __init__(self, dims, expansion_factor=1., dropout=0., use_single_layer=False,
                 out_dims=0, use_act=True,
                 save_h_neigh_grad=False):
        super().__init__()

        self.h_v = None
        self.h_neigh = None
        self.save_grad = save_h_neigh_grad

        self.use_single_layer = use_single_layer
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.use_act = use_act

        out_dims = dims if out_dims == 0 else out_dims

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, out_dims)
            self.detached_linear_0 = nn.Linear(dims, out_dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.detached_linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), out_dims)

        self.reset_parameters()

    def reset_parameters(self, init_type='model', gain=1.0):
        if init_type == 'model':
            self.linear_0.reset_parameters()
            if not self.use_single_layer:
                self.linear_1.reset_parameters()
        elif init_type == 'sampler':
            init.xavier_uniform_(self.linear_0.weight, gain=gain)
            init.zeros_(self.linear_0.bias)
            if not self.use_single_layer:
                init.xavier_uniform_(self.linear_1.weight, gain=gain)
                init.zeros_(self.linear_1.bias)
        elif init_type == 'model_zero':
            init.kaiming_uniform_(self.linear_0.weight, a=math.sqrt(5))
            init.zeros_(self.linear_0.bias)
            if not self.use_single_layer:
                init.kaiming_uniform_(self.linear_1.weight, a=math.sqrt(5))
                init.zeros_(self.linear_1.bias)
        else:
            raise NotImplementedError

    def sample_loss(self, log_prob):
        grad_h_neigh = self.h_neigh.grad.detach()

        self.detached_linear_0.load_state_dict(self.linear_0.state_dict())
        for para in self.detached_linear_0.parameters():
            para.requires_grad = False
        h_neigh = self.detached_linear_0(log_prob.unsqueeze(1) * self.h_v.detach())

        batch_loss = torch.bmm(grad_h_neigh.view(grad_h_neigh.shape[0], 1, -1),
                               h_neigh.view(h_neigh.shape[0], -1, 1))

        # none negative node, bad performance
        # batch_size = batch_loss.shape[0] // 3 * 2
        # batch_loss = batch_loss[:batch_size]

        return batch_loss

    def forward(self, x):
        if x.shape[-1] == 0:
            return x

        if self.save_grad:
            self.h_v = x

        x = self.linear_0(x)

        if self.save_grad:
            self.h_neigh = x
            if self.training: self.h_neigh.retain_grad()

        if self.use_act:
            x = F.gelu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.use_single_layer:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """

    def __init__(self, num_neighbor, dim_feat,
                 token_expansion_factor=0.5,
                 channel_expansion_factor=4.,
                 dropout=0.,
                 save_h_neigh_grad=False):
        super().__init__()

        self.token_layernorm = nn.LayerNorm(dim_feat)
        self.token_forward = FeedForward(num_neighbor, token_expansion_factor, dropout,
                                         save_h_neigh_grad=save_h_neigh_grad)

        self.channel_layernorm = nn.LayerNorm(dim_feat)
        self.channel_forward = FeedForward(dim_feat, channel_expansion_factor, dropout)

    def reset_parameters(self, init_type='model', gain=1.0):
        self.token_layernorm.reset_parameters()
        self.token_forward.reset_parameters(init_type, gain)

        self.channel_layernorm.reset_parameters()
        self.channel_forward.reset_parameters(init_type, gain)

    def sample_loss(self, log_prob):
        return self.token_forward.sample_loss(log_prob)

    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MixerAggregator(torch.nn.Module):

    def __init__(self, num_neighbor, dim_node_feat, dim_edge_feat, dim_time, dim_out, dropout=0.0,
                 time_encoder_type='fixed', save_h_neigh_grad=False):
        super(MixerAggregator, self).__init__()

        self.num_neighbor = num_neighbor
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out

        if time_encoder_type == 'learnable':
            self.time_encoder = LearnableTimeEncoder(dim_time)
        elif time_encoder_type == 'fixed':
            self.time_encoder = FixedTimeEncoder(dim_time)
        else:
            raise NotImplementedError

        self.mixer = MixerBlock(num_neighbor, dim_node_feat + dim_edge_feat + dim_time,
                                dropout=dropout, save_h_neigh_grad=save_h_neigh_grad)

        self.layer_norm = torch.nn.LayerNorm(dim_node_feat + dim_edge_feat + dim_time)
        self.mlp_out = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)

    def sample_loss(self, log_prob):
        return self.mixer.sample_loss(log_prob)

    def forward(self, block):
        edge_time_feat = self.time_encoder((block.root_ts.unsqueeze(-1) - block.neighbor_ts).flatten())

        neighbor_node_feature = block.neighbor_node_feature.view(
            block.neighbor_node_feature.shape[0] * block.neighbor_node_feature.shape[1],
            block.neighbor_node_feature.shape[2])
        neighbor_edge_feature = block.neighbor_edge_feature.view(
            block.neighbor_edge_feature.shape[0] * block.neighbor_edge_feature.shape[1],
            block.neighbor_edge_feature.shape[2])

        feats = torch.cat([neighbor_node_feature, neighbor_edge_feature, edge_time_feat], dim=1)
        feats = feats.view(-1, self.num_neighbor, feats.shape[-1])

        feats = self.mixer(feats)

        h_out = self.layer_norm(feats)
        h_out = torch.mean(h_out, dim=1)
        h_out = self.mlp_out(h_out)

        return h_out


class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0] // (neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class SelfNorm(torch.nn.Module):
    def __init__(self, num_scope, dropout=0., eps=1e-5):
        super(SelfNorm, self).__init__()

        self.norm = nn.LayerNorm(num_scope)
        self.linear = nn.Linear(num_scope, 1)
        self.dropout = nn.Dropout(p=dropout)

        # self.weight = nn.Parameter(torch.tensor([0.1]))
        # self.bias = nn.Parameter(torch.tensor([0.]))
        # self.weight = torch.tensor([1e-1])
        # self.bias = torch.tensor([0.])
        # self.eps = eps

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.linear.reset_parameters()

        # self.weight = nn.Parameter(torch.tensor([1.]))
        # self.bias = nn.Parameter(torch.tensor([0.]))


    def forward(self, x, neigh_mask):
        d = x.unsqueeze(dim=1).repeat(1, 25, 1)
        r = torch.zeros_like(d)  # (1800, 25, 25)
        r[neigh_mask] = d[neigh_mask]

        # neigh_count = neigh_mask.sum(dim=2)
        # mean = r.sum(dim=2) / neigh_count
        # var = torch.square(r - mean.unsqueeze(dim=2)).sum(dim=2) / neigh_count
        # x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # x = self.weight * x_norm + self.bias

        r = self.norm(r)
        x = self.linear(r).squeeze(-1)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class AttentionDecoder(torch.nn.Module):

    def __init__(self, dim_neigh_encode, dim_root_encode, att_type, dim_embed=100):
        super(AttentionDecoder, self).__init__()

        att_type = att_type.lower()
        assert att_type in ['transformer', 'gat_v1', 'gat_v2']
        self.att_type = att_type
        self.dim_embed = dim_embed
        self.dim_neigh_encode = dim_neigh_encode
        self.dim_root_encode = dim_root_encode

        self.w_q = nn.Linear(dim_root_encode, dim_embed, bias=True)  # bias is important for attention
        self.w_k = nn.Linear(dim_neigh_encode, dim_embed, bias=True)
        self.att_act = nn.LeakyReLU(0.2)
        self.att = nn.Parameter(torch.empty(dim_embed, 1))

        self.reset_parameters()

    def reset_parameters(self, gain=1.0, bias=False):
        """
        For weight init, xavier is better than kaiming for attention, linear vise versa
        """
        nn.init.xavier_uniform_(self.w_q.weight, gain)
        nn.init.xavier_uniform_(self.w_q.weight, gain)
        nn.init.xavier_uniform_(self.att, gain)

        # nn.init.kaiming_uniform_(self.w_q.weight, 0.2, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.w_q.weight, 0.2, nonlinearity='leaky_relu')
        # nn.init.kaiming_uniform_(self.att, 0.2, nonlinearity='leaky_relu')

        """
        For bias init, zero is better for mixer+att, uniform is better for att only
        """
        if bias:
            bound = 1 / math.sqrt(self.dim_root_encode)
            init.uniform_(self.w_q.bias, -bound, bound)
            bound = 1 / math.sqrt(self.dim_neigh_encode)
            init.uniform_(self.w_k.bias, -bound, bound)
        else:
            nn.init.zeros_(self.w_q.bias)
            nn.init.zeros_(self.w_k.bias)

    def forward(self, neigh_encode, root_encode):
        h_q = self.w_q(root_encode).view(root_encode.shape[0], 1, self.dim_embed)
        h_k = self.w_k(neigh_encode).view(root_encode.shape[0], -1, self.dim_embed)
        h = h_q + h_k

        if self.att_type == 'transformer':  # good for mixer
            alpha = self.att_act(torch.sum(h_q * h_k, dim=-1))
        elif self.att_type == 'gat_v1':
            alpha = self.att_act(h @ self.att)
        elif self.att_type == 'gat_v2':  # good for tgat
            alpha = self.att_act(h) @ self.att
        else:
            raise NotImplementedError

        return F.softmax(alpha, dim=1)


class LinkMapper(torch.nn.Module):
    """
    Mapping Link Encoding to Sample Probability
    (num_roots, num_scope, dim) -> (num_roots, num_scope)
    """

    def __init__(self, link_encode_dims, root_encode_dims, num_scope,
                 feat_norm=False, neigh_norm=False, dropout=0., decoder_type='transformer',
                 enable_mixer=True, init_gain=1.0, unif_bias=False):
        super(LinkMapper, self).__init__()

        self.decoder_type = decoder_type.lower()

        if enable_mixer:
            self.mixer = MixerBlock(num_scope, link_encode_dims, dropout=dropout,
                                    token_expansion_factor=0.2, channel_expansion_factor=0.2)
        if self.decoder_type == 'linear':
            self.decoder = nn.Sequential(
                FeedForward(link_encode_dims, out_dims=1, dropout=dropout, use_single_layer=True,
                            use_act=True),
                nn.Softmax(dim=1),
            )
        else:
            self.decoder = AttentionDecoder(link_encode_dims, root_encode_dims, self.decoder_type, dim_embed=100)

        if feat_norm:
            self.feat_norm = nn.LayerNorm(link_encode_dims)
            self.feat_norm_root = nn.LayerNorm(root_encode_dims)
        self.neigh_norm = neigh_norm

        self.reset_parameters(init_gain, unif_bias)

    def reset_parameters(self, gain, bias):
        if hasattr(self, 'mixer'):
            self.mixer.reset_parameters(init_type='model')
        if self.decoder_type == 'linear':
            self.decoder[0].reset_parameters(init_type='model')
        else:
            self.decoder.reset_parameters(gain=gain, bias=bias)

    def forward(self, x, x_root):
        if hasattr(self, 'feat_norm'):
            x = self.feat_norm(x)
            x_root = self.feat_norm_root(x_root)

        if hasattr(self, 'mixer'):
            x = self.mixer(x)  # significant improvement in GraphMixer

        if self.neigh_norm:
            x = F.normalize(x, p=2, dim=1)  # works alright

        if self.decoder_type == 'linear':
            x = self.decoder(x)
        else:
            x = self.decoder(x, x_root)

        """
        Trials:
        x = F.gelu(x)  # bad performance, very slow convergence
        x = self.feat_norm(x)  # useless

        x = x + self.self_norm(x, *args)  # bad
        x = self.neigh_norm(x)  # very bad performance
        x = F.normalize(x, p=2, dim=1)  # bad at here

        x = F.softmax(x, dim=1)  # softmax is important
        x = F.relu(x) + 10e-10  # very bad
        x = F.dropout(x, p=0.1, training=self.training) + 1e-10  # pretty bad
        
        x = self.decoder(x, x.mean(dim=1)) # a little bit worse 
        """
        return x.squeeze()
