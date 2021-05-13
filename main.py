#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/5/10 10:37
# @Author : Jinkai Zheng
# @Email : zhengjinkai3@qq.com

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as tordata
import torch.nn.functional as F

import os
import time
from datetime import datetime
import os.path as osp
import numpy as np


from lib.non_parametric_classifier import NonParametricClassifier
from lib.criterion import Criterion
from lib.ans_discovery import ANsDiscovery
from lib.utils import AverageMeter, time_progress, adjust_learning_rate

from packages import lr_policy
from logger import log

from GaitSet.model.utils.data_loader import load_data
from GaitSet.model.utils.evaluator import evaluation
from GaitSet.model.network import SetNet_OU
from GaitSet.config import conf_CASIA, conf_OULP

from configs import parser_argument
from collect_fn import train_collate_fn, test_collate_fn

def require_args():
    
    cfg.add_argument('--max-epoch', default=200, type=int,
                        help='max epoch per round. (default: 200)')
    cfg.add_argument('--max-round', default=4, type=int,
                        help='max iteration, including initialisation one. '
                             '(default: 5)')
    cfg.add_argument('--source', default='casia-b', type=str,
                        help='source dataset to be used. (default: casia-b)')
    cfg.add_argument('--target', default='oulp', type=str,
                        help='target dataset to be used. (default: oulp)')


def main():

    log.info('Start to declare training variables')
    cfg.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0.  # best test accuracy
    best_model_dir = osp.join('outputs/TraND', cfg.log_name, 'best_model', )
    os.makedirs(best_model_dir, exist_ok=True)

    if cfg.source == "casia-b":
        config = conf_CASIA
    elif cfg.source == "oulp":
        config = conf_OULP
    else:
        raise Warning("Please check your source/target dataset, current dataset is not casia-b or oulp.")

    # config for source dataset
    WORK_PATH = config['WORK_PATH']
    data_config = config['data']
    model_config = config['model']
    model_name = model_config["model_name"]
    batch_size = int(np.prod(model_config['batch_size']))
    save_name = '_'.join(map(str, [
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))
    os.chdir(WORK_PATH)

    if cfg.target == "casia-b":
        config = conf_CASIA
    elif cfg.target == "oulp":
        config = conf_OULP
    else:
        raise Warning("Please check your source/target dataset, current dataset is not casia-b or oulp.")

    # config for target dataset
    model_config = config['model']
    num_workers = model_config['num_workers']
    hidden_dim = model_config['hidden_dim']

    log.info('Start to prepare data')

    trainset, testset = load_data(**config['data'], cache=True)
    trainloader = tordata.DataLoader(dataset=trainset, batch_size=128,
                                     collate_fn=train_collate_fn, num_workers=num_workers)
    testloader = tordata.DataLoader(dataset=testset, batch_size=64,
                                    collate_fn=test_collate_fn, num_workers=num_workers)
    # cheat labels are only used to compute the neighbourhoods consistency
    cheat_labels = torch.tensor([int(i) for i in trainset.label]).long().to(device)
    ntrain, ntest = len(trainset), len(testset)
    log.info('Totally got %d training and %d testing samples' % (ntrain, ntest))

    log.info('Start to build model')
    net = SetNet_OU(hidden_dim).float()
    ANs_discovery = ANsDiscovery(ntrain)
    criterion = Criterion()

    # data parallel
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(len(cfg.gpus.split(','))))
        cudnn.benchmark = True

    net, ANs_discovery, criterion = (net.to(device), ANs_discovery.to(device), criterion.to(device))

    optimizer = optim.Adam([{'params': net.parameters()},], lr=cfg.base_lr)

    if cfg.load_pretrained:
        load(net, optimizer, WORK_PATH, model_name, save_name, restore_iter=100000)

    source_memory, _, _, _ = gait_transform(net, trainloader)

    npc = NonParametricClassifier(source_memory, cfg.low_dim, ntrain, cfg.npc_temperature, cfg.npc_momentum)
    npc = npc.to(device)

    round = cfg.start_round

    while (round <= cfg.max_round):

        log.info('Start training at %d/%d round' % (round, cfg.max_round))

        params = torch.tensor([0.1, 0.5])
        ANs_discovery.update(round, npc, params, cheat_labels)
        log.info('ANs consistency at %d round is %.2f%%' % (round, ANs_discovery.consistency * 100))

        epoch = 0
        lr = cfg.base_lr
        lr_handler = lr_policy.get(cfg.lr_policy, instant=True)

        while lr > 0 and epoch < cfg.max_epoch + 1:

            # get learning rate according to current epoch
            lr = lr_handler.update(epoch)

            if epoch != 0:
                train(round, epoch, net, trainloader, optimizer, npc, criterion, ANs_discovery, lr)

            if epoch % 5 == 0:
                log.info('Start to evaluate...')
                log.info('Transforming...')
                time = datetime.now()
                test = gait_transform(net, testloader)
                log.info('Evaluating...')
                acc = evaluation(test, config['data'])
                print('Evaluation complete. Cost:', datetime.now() - time)
                acc_vis(acc)
                if np.mean(acc[0, :, :, 0]) > best_acc:
                    best_acc = np.mean(acc[0, :, :, 0])
                    log.info('Saving the best model and optimizer...')
                    save(net, optimizer, best_model_dir, WORK_PATH)

            epoch += 1
        round += 1

    log.info("\nThe best result is...")
    net.load_state_dict(torch.load(osp.join(best_model_dir, 'best_encoder.ptm')))
    test = gait_transform(net, testloader)
    acc = evaluation(test, config['data'])
    acc_vis(acc)

def acc_vis(acc):
    if acc.shape[0] == 3:
        # Print rank-1 accuracy of the best model
        for i in range(1):
            log.info('===Rank-%d (Include identical-view cases)===' % (i + 1))
            log.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))

        # Print rank-1 accuracy of the best model，excluding identical-view cases
        for i in range(1):
            log.info('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            log.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))

        # Print rank-1 accuracy of the best model (Each Angle)
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            log.info('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            log.info('NM: %s', str(de_diag(acc[0, :, :, i], True)))
            log.info('BG: %s', str(de_diag(acc[1, :, :, i], True)))
            log.info('CL: %s', str(de_diag(acc[2, :, :, i], True)))
    elif acc.shape[0] == 1:
        # Print rank-1 accuracy of the best model
        for i in range(1):
            log.info('===Rank-%d (Include identical-view cases)===' % (i + 1))
            log.info('NM: %.3f' % (
                np.mean(acc[0, :, :, i])))

        # Print rank-1 accuracy of the best model，excluding identical-view cases
        for i in range(1):
            log.info('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            log.info('NM: %.3f' % (
                de_diag(acc[0, :, :, i])))

        # Print rank-1 accuracy of the best model (Each Angle)
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            log.info('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            log.info('NM: %s', str(de_diag(acc[0, :, :, i], True)))

def train(round, epoch, net, trainloader, optimizer, npc, criterion, ANs_discovery, lr):

    # tracking variables
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch the model to train mode
    net.train()
    # adjust learning rate
    adjust_learning_rate(optimizer, lr)

    end = time.time()
    optimizer.zero_grad()
    for batch_idx, (seq, view, seq_type, label, batch_frame, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)
        indexes = torch.LongTensor(indexes).to(cfg.device)

        for i in range(len(seq)):
            seq[i] = np2var(seq[i]).float()
        if batch_frame is not None:
            batch_frame = np2var(batch_frame).int()

        feature, _ = net(*seq, batch_frame)

        n, num_bin, _ = feature.size()
        feature = feature.view(n, -1)
        feature = F.normalize(feature, p=2, dim=1)
        params = torch.tensor([0.1, 0.5])

        outputs = npc(feature, indexes, params)
        loss, loss_inst, loss_ans = criterion(outputs, indexes, ANs_discovery)

        loss.backward()
        train_loss.update(loss.item() * cfg.iter_size, feature.size(0))

        if batch_idx % cfg.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % cfg.display_freq != 0:
            continue

        elapsed_time, estimated_time = time_progress(batch_idx + 1, len(trainloader), batch_time.sum)
        log.info('Round: {round} Epoch: {epoch}/{tot_epochs} '
                 'Progress: {elps_iters}/{tot_iters} ({elps_time}/{est_time}) '
                 'Data: {data_time.avg:.3f} LR: {learning_rate:.8f} '
                 'Loss: {train_loss.val:.5f} ({train_loss.avg:.5f}) '
                 'Loss_inst: {loss_inst:.5f} Loss_ans: {loss_ans:.2f}'
            .format(round=round, epoch=epoch, tot_epochs=cfg.max_epoch,
                    elps_iters=batch_idx, tot_iters=len(trainloader),
                    elps_time=elapsed_time, est_time=estimated_time,
                    data_time=data_time, learning_rate=lr,
                    train_loss=train_loss, loss_inst=loss_inst.item(), loss_ans=loss_ans.item()))

def gait_transform(net, testloader):
    net.eval()

    feature_list = list()
    view_list = list()
    seq_type_list = list()
    label_list = list()

    for i, x in enumerate(testloader):
        seq, view, seq_type, label, batch_frame, index = x
        for j in range(len(seq)):
            seq[j] = np2var(seq[j]).float()
        if batch_frame is not None:
            batch_frame = np2var(batch_frame).int()

        feature, _ = net(*seq, batch_frame)

        n, num_bin, _ = feature.size()
        feature = feature.view(n, -1)
        feature = F.normalize(feature, p=2, dim=1)

        feature_list.append(feature.data.cpu().numpy())
        view_list += view
        seq_type_list += seq_type
        label_list += label

    return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

def save(net, optimizer, best_model_dir, WORK_PATH):
    if not osp.exists(best_model_dir): os.makedirs(best_model_dir, exist_ok=True)
    os.chdir("../../")
    torch.save(net.state_dict(), osp.join(best_model_dir, 'best_encoder.ptm'))
    torch.save(optimizer.state_dict(), osp.join(best_model_dir, 'best_optimizer.ptm'))
    os.chdir(WORK_PATH)

def load(net, optimizer, WORK_PATH, model_name, save_name, restore_iter):
    os.chdir("../../")
    net.load_state_dict(torch.load(osp.join(
        WORK_PATH, 'checkpoint', model_name,
        '{}-{:0>5}-encoder.ptm'.format(save_name, restore_iter))))
    optimizer.load_state_dict(torch.load(osp.join(
        WORK_PATH, 'checkpoint', model_name,
        '{}-{:0>5}-optimizer.ptm'.format(save_name, restore_iter))))
    os.chdir(WORK_PATH)

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1)
    result = result / (result.shape[0]-1.0)
    if not each_angle:
        result = np.mean(result)
    return result

def np2var(x):
    return ts2var(torch.from_numpy(x))

def ts2var(x):
    return torch.autograd.Variable(x).cuda()


if __name__ == '__main__':
    cfg = parser_argument()
    main()
