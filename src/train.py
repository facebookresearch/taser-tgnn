#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--root_path', type=str, default='DATA', help='dataset root path')
parser.add_argument('--config', type=str, help='path to config file')
parser.add_argument('--gpu', type=str, default="0", help='which GPU to use')
parser.add_argument('--eval_neg_samples', type=int, default=49, help='how many negative samples to use at inference.')

parser.add_argument('--cached_ratio', type=float, default=0.3, help='the ratio of gpu cached edge feature')
parser.add_argument('--cache', action='store_true', help='cache edge features on device')
parser.add_argument('--pure_gpu', action='store_true', help='put all edge features on device, disable cache')
parser.add_argument('--print_cache_hit_rate', action='store_true', help='print cache hit rate each epoch. Note: this will slowdown performance.')

parser.add_argument('--tb_log_prefix', type=str, default='', help='prefix for the tb logging data.')

parser.add_argument('--profile', action='store_true', help='whether to profile.')
parser.add_argument('--profile_prefix', default='log_profile/', help='prefix for the profiling data.')

parser.add_argument('--edge_feature_access_fn', default='', help='prefix to store the edge feature access pattern per epoch')

parser.add_argument('--override_epoch', type=int, default=0, help='override epoch in config.')
parser.add_argument('--override_lr', type=float, default=-1, help='override learning rate in config.')
parser.add_argument('--override_order', type=str, default='', help='override training order in config.')
parser.add_argument('--override_scope', type=int, default=0, help='override sampling scope in config.')
parser.add_argument('--override_neighbor', type=int, default=0, help='override sampling neighbors in config.')

parser.add_argument('--gradient_option', type=str, default='none', choices=["none", "unbiased"])

parser.add_argument('--no_time', action='store_true', help='do not record time (avoid extra cuda synchronization cost).')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import globals

import yaml
from utils import *
from model import TGNN
from model import AdaptSampler
from dataloader import DataLoader
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score

if not args.no_time:
    globals.timer.set_enable()

@torch.no_grad()
def eval(model, dataloader):
    model.eval()
    if dataloader.sampler is not None:
        dataloader.sampler.eval()
    aps = list()
    mrrs = list()
    while not dataloader.epoch_end:
        blocks = dataloader.get_blocks()
        pred_pos, pred_neg = model(blocks)
        y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
        y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
        aps.append(average_precision_score(y_true, y_pred))
        mrrs.append(torch.reciprocal(
            torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(blocks[-1].num_neg_dst, -1), dim=0) + 1).type(
            torch.float))
    dataloader.reset()
    ap = float(torch.tensor(aps).mean())
    mrr = float(torch.cat(mrrs).mean())
    return ap, mrr


config = yaml.safe_load(open(args.config, 'r'))

if args.override_epoch > 0:
    config['train'][0]['epoch'] = args.override_epoch
if args.override_lr > 0:
    config['train'][0]['lr'] = args.override_lr
if args.override_order != '':
    config['train'][0]['order'] = args.override_order
if args.override_scope > 0:
    fanout = config['scope'][0]['neighbor']
    for i in range(len(fanout)):
        fanout[i] = args.override_scope
if args.override_neighbor > 0:
    config['sample'][0]['neighbor'] = args.override_neighbor

"""Logger"""
path_saver = 'models/{}_{}_{}.pkl'.format(args.data, args.config.split('/')[1].split('.')[0],
                                          time.strftime('%m-%d %H:%M:%S'))
path = os.path.dirname(path_saver)
os.makedirs(path, exist_ok=True)

if args.tb_log_prefix != '':
    tb_path_saver = 'log_tb/{}{}_{}_{}'.format(args.tb_log_prefix, args.data,
                                               args.config.split('/')[1].split('.')[0],
                                               time.strftime('%m-%d %H:%M:%S'))
    os.makedirs('log_tb/', exist_ok=True)
    writer = SummaryWriter(log_dir=tb_path_saver)

if args.edge_feature_access_fn != '':
    os.makedirs('../log_cache/', exist_ok=True)
    efeat_access_path_saver = 'log_cache/{}'.format(args.edge_feature_access_fn)

profile_path_saver = '{}{}_{}_{}'.format(args.profile_prefix, args.data,
                                         args.config.split('/')[1].split('.')[0],
                                         time.strftime('%m-%d %H:%M:%S'))
path = os.path.dirname(profile_path_saver)
os.makedirs(path, exist_ok=True)

"""Data"""
g, edges, nfeat, efeat = load_data(args.data, args.root_path)
if efeat is not None and efeat.dtype == torch.bool:
    efeat = efeat.to(torch.int8)
if nfeat is not None and nfeat.dtype == torch.bool:
    nfeat = nfeat.to(torch.int8)
dim_edge_feat = efeat.shape[-1] if efeat is not None else 0
dim_node_feat = nfeat.shape[-1] if nfeat is not None else 0

"""Model"""
device = 'cuda'
sampler = None
params = []
if config['sample'][0]['type'] == 'adapt':
    sampler = AdaptSampler(config, dim_edge_feat, dim_node_feat).to(device)
    params.append({
        'params': sampler.parameters(),
        'lr': config['sample'][0]['lr'],
        'weight_decay': float(config['sample'][0]['weight_decay']),
        # 'betas': (0.99, 0.9999)
    })
model = TGNN(config, dim_node_feat, dim_edge_feat).to(device)
params.append({
    'params': model.parameters(),
    'lr': config['train'][0]['lr']
})
optimizer = torch.optim.Adam(params)
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

"""Loader"""
train_loader = DataLoader(g, config['scope'][0]['neighbor'],
                          edges['train_src'], edges['train_dst'], edges['train_time'], edges['neg_dst'],
                          nfeat, efeat, config['train'][0]['batch_size'],
                          sampler=sampler,
                          device=device, mode='train',
                          type_sample=config['scope'][0]['strategy'],
                          order=config['train'][0]['order'],
                          # edge_deg=edges['train_deg'],
                          cached_ratio=args.cached_ratio, enable_cache=args.cache, pure_gpu=args.pure_gpu)
val_loader = DataLoader(g, config['scope'][0]['neighbor'],
                        edges['val_src'], edges['val_dst'], edges['val_time'], edges['neg_dst'],
                        nfeat, efeat, config['eval'][0]['batch_size'],
                        sampler=sampler,
                        device=device, mode='val',
                        eval_neg_dst_nid=edges['val_neg_dst'],
                        type_sample=config['scope'][0]['strategy'],
                        enable_cache=False, pure_gpu=args.pure_gpu)
test_loader = DataLoader(g, config['scope'][0]['neighbor'],
                         edges['test_src'], edges['test_dst'], edges['test_time'], edges['neg_dst'],
                         nfeat, efeat, config['eval'][0]['batch_size'],
                         sampler=sampler,  # load from dict
                         device=device, mode='test',
                         eval_neg_dst_nid=edges['test_neg_dst'],
                         type_sample=config['scope'][0]['strategy'],
                         enable_cache=False, pure_gpu=args.pure_gpu)

if args.edge_feature_access_fn != '':
    efeat_access_freq = list()

best_ap = 0
best_e = 0
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=50, warmup=50, active=20, skip_first=100, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path_saver)
) if args.profile else nullcontext() as profiler:
    for e in range(config['train'][0]['epoch']):
        print('Epoch {:d}:'.format(e))

        if args.edge_feature_access_fn != '':
            efeat_access_freq.append(torch.zeros(efeat.shape[0], dtype=torch.int32, device='cpu'))

        # training
        model.train()
        if sampler is not None:
            sampler.train()
        total_loss = 0
        total_sample_loss = 0

        if args.no_time: t_s = time.time()
        if args.print_cache_hit_rate:
            hit_count = 0
            miss_count = 0
        while not train_loader.epoch_end:
            blocks = train_loader.get_blocks(log_cache_hit_miss=args.print_cache_hit_rate)
            if args.print_cache_hit_rate:
                for block in blocks:
                    hit_count += block.cache_hit_count
                    miss_count += block.cache_miss_count

            if args.edge_feature_access_fn != '':
                with torch.no_grad():
                    for b in blocks:
                        access = b.neighbor_eid.flatten().detach().cpu()
                        value = torch.ones_like(access, dtype=torch.int32)
                        efeat_access_freq[-1].put_(access, value, accumulate=True)

            globals.timer.start_train()
            optimizer.zero_grad()
            pred_pos, pred_neg = model(blocks)
            loss_pos = criterion(pred_pos, torch.ones_like(pred_pos))
            loss = loss_pos.mean()
            loss += criterion(pred_neg, torch.zeros_like(pred_neg)).mean()
            loss.backward()

            if config['sample'][0]['type'] == 'adapt':
                log_prob = sampler.get_log_prob()
                sample_loss = model.sample_loss(log_prob)
                if config['train'][0]['order'].startswith('gradient') and args.gradient_option == 'unbiased':
                    """ Unbiased loss:
                    n = train_loader.src_nid.size(0)
                    new_loss_i = loss_i * (1/p) * (1/n) = loss_i / (n*p)
                    Exp(loss_i) = sum(loss_i * (1/n))
                    Exp (new_loss_i) = sum(loss_i * (1/p) * (1/n) * p) = Exp(loss_i)
                    """
                    src_prob = train_loader.root_prob[blocks[-1].gradient_idx]
                    if len(blocks) == 1:
                        # GraphMixer's one layer model
                        sample_loss[:-blocks[-1].neg_dst_size] /= train_loader.src_nid.size(0) * src_prob.repeat(2)
                        # sample_loss[:] /= train_loader.src_nid.size(0) * src_prob.repeat(2)  # without neg edges
                    elif len(blocks) == 2:
                        # TGAT's two layer model:
                        # 1800 * 10 + 1800, 0 ~ 12000 & 18000 ~ 19200
                        sample_loss[:blocks[-1].src_size * 2 * sampler.num_sample] /= \
                            train_loader.src_nid.size(0) * src_prob.unsqueeze(-1).repeat(2, sampler.num_sample).flatten()
                        sample_loss[-blocks[-1].n: -blocks[-1].neg_dst_size] /= \
                            train_loader.src_nid.size(0) * src_prob.repeat(2)
                    else:
                        raise NotImplementedError
                sample_loss = sample_loss.mean()
                sample_loss.backward()
                # torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1, norm_type=2, error_if_nonfinite=True)
            else:
                sample_loss = 0

            optimizer.step()
            globals.timer.end_train()

            if args.profile:
                profiler.step()

            with torch.no_grad():
                total_loss += float(loss) * config['train'][0]['batch_size']
                total_sample_loss += float(sample_loss) * config['train'][0]['batch_size']
                if config['train'][0]['order'].startswith('gradient'):
                    weights = torch.special.expit(pred_pos)
                    train_loader.update_gradient(blocks[-1].gradient_idx, weights)
        if args.print_cache_hit_rate:
            oracle_cache_hit_rate = train_loader.reset(log_cache_hit_miss=True)
        else:
            train_loader.reset()

        if args.no_time:
            torch.cuda.synchronize()
            t_rough = time.time() - t_s

        ap = mrr = 0.
        time_val = 0.
        if e >= config['eval'][0]['val_epoch']:
            globals.timer.start_val()
            ap, mrr = eval(model, val_loader)
            globals.timer.end_val()
            if ap > best_ap:
                best_e = e
                best_ap = ap
                param_dict = {'model': model.state_dict()}
                if sampler is not None:
                    param_dict['sampler'] = sampler.state_dict()
                torch.save(param_dict, path_saver)

        if args.tb_log_prefix != '':
            writer.add_scalar(tag='Loss/Train', scalar_value=total_loss, global_step=e)
            writer.add_scalar(tag='Loss/Sample', scalar_value=total_sample_loss, global_step=e)
            writer.add_scalar(tag='AP/Val', scalar_value=ap, global_step=e)
            writer.add_scalar(tag='MRR/Val', scalar_value=mrr, global_step=e)
        if config['sample'][0]['type'] == 'adapt':
            print('\ttrain loss:{:.4f}  sample loss:{:.4f}  val ap:{:4f}  val mrr:{:4f}'.format(total_loss, total_sample_loss, ap, mrr))
        else:
            print('\ttrain loss:{:.4f}  val ap:{:4f}  val mrr:{:4f}'.format(total_loss, ap, mrr))
        if args.no_time:
            print('\trough train time: {:.2f}s'.format(t_rough))
        if args.print_cache_hit_rate:
            print('\tcache hit rate: {:.2f}%  oracle hit rate: {:.2f}%'.format(hit_count / (hit_count + miss_count) * 100, oracle_cache_hit_rate * 100))
        else:
            globals.timer.print(prefix='\t')
            globals.timer.reset()

if args.tb_log_prefix != '':
    writer.close()

if args.edge_feature_access_fn != '':
    torch.save(efeat_access_freq, efeat_access_path_saver)

print('Loading model at epoch {} with val AP {:4f}...'.format(best_e, best_ap))
param_dict = torch.load(path_saver)
model.load_state_dict(param_dict['model'])
if sampler is not None:
    sampler.load_state_dict(param_dict['sampler'])
ap, mrr = eval(model, test_loader)
print('\ttest AP:{:4f}  test MRR:{:4f}'.format(ap, mrr))
