from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate
from libs.models.STM import STM
from libs.models.Att import Att

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict
from random import shuffle

from options import OPTION as opt

MAX_FLT = 1e6

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='1', type=str, help='set gpu id to train the network, split with comma')
    return parser.parse_args()


def main():
    with torch.cuda.device(1):
        start_epoch = 0
        random.seed(0)

        args = parse_args()
        # Use GPU
        use_gpu = torch.cuda.is_available() and (args.gpu != '' or int(opt.gpu_id)) >= 0
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id)
        gpu_ids = [int(val) for val in args.gpu.split(',')]

        if not os.path.isdir(opt.checkpoint):
            os.makedirs(opt.checkpoint)

        # Data
        print('==> Preparing dataset')

        input_dim = opt.input_size

        train_transformer = TrainTransform(size=input_dim)
        test_transformer = TestTransform(size=input_dim)


        try:
            if isinstance(opt.trainset, list):
                datalist = []
                for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                    ds = DATA_CONTAINER[dataset](
                        train=True,
                        sampled_frames=opt.sampled_frames,
                        transform=train_transformer,
                        max_skip=max_skip,
                        samples_per_video=opt.samples_per_video
                    )
                    datalist += [ds] * freq

                trainset = data.ConcatDataset(datalist)

            else:
                max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
                trainset = DATA_CONTAINER[opt.trainset](
                    train=True,
                    sampled_frames=opt.sampled_frames,
                    transform=train_transformer,
                    max_skip=max_skip,
                    samples_per_video=opt.samples_per_video
                )
        except KeyError as ke:
            print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
            print(list(DATA_CONTAINER.keys()))
            exit()

        testset = DATA_CONTAINER[opt.valset](
            train=False,
            transform=test_transformer,
            samples_per_video=1
        )

        trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                      collate_fn=multibatch_collate_fn, drop_last=True)

        # Model
        print("==> creating model")

        net = STM(opt.keydim, opt.valdim, 'train',
                  mode=opt.mode, iou_threshold=opt.iou_threshold)
        net.eval()

        if use_gpu:
            net = net.cuda()

        att = Att(save_freq=opt.save_freq, keydim = opt.keydim, valdim=opt.valdim)
        att.eval()
        if use_gpu:
            att = att.cuda()


        print('    Total params need to train: %.2fM' % ((sum(p.numel() for p in net.Decoder.parameters())
        + sum(p.numel() for p in att.parameters()) ) / 1000000.0))

        # set training parameters
        for p in net.Encoder_M.parameters():
            p.requires_grad = False
        for p in net.Encoder_Q.parameters():
            p.requires_grad = False
        for p in net.KV_M_r4.parameters():
            p.requires_grad = True
        for p in net.KV_Q_r4.parameters():
            p.requires_grad = True
        for p in net.KV_M_r3.parameters():
            p.requires_grad = True
        for p in net.KV_Q_r3.parameters():
            p.requires_grad = True
        for p in net.Decoder.parameters():
            p.requires_grad = True

        for p in att.parameters():
            p.requires_grad = True



        criterion = None
        celoss = cross_entropy_loss

        if opt.loss == 'ce':
            criterion = celoss
        elif opt.loss == 'iou':
            criterion = mask_iou_loss
        elif opt.loss == 'both':
            criterion = lambda pred, target, obj: celoss(pred, target, obj) + mask_iou_loss(pred, target, obj)
        else:
            raise TypeError('unknown training loss %s' % opt.loss)

        if opt.solver == 'sgd':
            
            params = [{"params": net.parameters()},
                      {"params": att.parameters()}]
            optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                                  momentum=opt.momentum[0], weight_decay=opt.weight_decay)
            
        elif opt.solver == 'adam':

            params = [{"params": net.parameters(), "lr": opt.learning_rate},
                      {"params": att.parameters(),  "lr": opt.learning_rate}]
            optimizer = optim.Adam(params, betas=opt.momentum, weight_decay=opt.weight_decay)
        else:
            raise TypeError('unkown solver type %s' % opt.solver)

        # Resume
        title = 'STM'
        minloss = float('inf')

        opt.checkpoint_STM = osp.join(osp.join(opt.checkpoint,  opt.valset, opt.setting, 'STM'))
        opt.checkpoint_att = osp.join(osp.join(opt.checkpoint, opt.valset, opt.setting, 'ATT'))
        if not osp.exists(opt.checkpoint_STM):
            os.makedirs(opt.checkpoint_STM)
        if not osp.exists(opt.checkpoint_att):
            os.makedirs(opt.checkpoint_att)

        if opt.initial_STM:
            print('==> Resuming from checkpoint {}'.format(opt.initial_STM))
            assert os.path.isfile(opt.initial_STM), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.initial_STM)
            state = checkpoint['state_dict']
            net.load_param(state)
        elif opt.resume_STM:
            # Load checkpoint.
            print('==> Resuming from pretrained {}'.format(opt.resume_STM))
            assert os.path.isfile(opt.resume_STM), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.resume_STM)
            net.load_state_dict(checkpoint['state_dict'],strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])

        logger = Logger(os.path.join(opt.checkpoint, opt.mode + '_log.txt'), resume=True)

        if opt.resume_ATT:
            # Load checkpoint.
            print('==> Resuming from checkpoint {}'.format(opt.resume_ATT))
            assert os.path.isfile(opt.resume_ATT), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(opt.resume_ATT)
            minloss = checkpoint['minloss']
            start_epoch = checkpoint['epoch']
            att.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            skips = checkpoint['max_skip']
            try:
                if isinstance(skips, list):
                    for idx, skip in enumerate(skips):
                        trainloader.dataset.datasets[idx].set_max_skip(skip)
                else:
                    trainloader.dataset.set_max_skip(skip)
            except:
                print('[Warning] Initializing max skip fail')


        logger.set_items(['Epoch', 'LR', 'Train Loss'])

        # Train and val
        for epoch in range(start_epoch):
            adjust_learning_rate(optimizer, epoch, opt)

        for epoch in range(start_epoch, opt.epochs):

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
            adjust_learning_rate(optimizer, epoch, opt)

            net.phase = 'train'

            train_loss = train(trainloader,
                               model=net,
                               Att_model=att,
                               criterion=criterion,
                               optimizer=optimizer,
                               epoch=epoch,
                               use_cuda=use_gpu,
                               iter_size=opt.iter_size,
                               mode=opt.mode,
                               threshold=opt.iou_threshold)

            # append logger file
            logger.log(epoch + 1, opt.learning_rate, train_loss)

            # adjust max skip
            if (epoch + 1) % opt.epochs_per_increment == 0:
                if isinstance(trainloader.dataset, data.ConcatDataset):
                    for dataset in trainloader.dataset.datasets:
                        dataset.increase_max_skip()
                else:
                    trainloader.dataset.increase_max_skip()

            # save model
            is_best = train_loss <= minloss
            minloss = min(minloss, train_loss)
            skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
                if isinstance(trainloader.dataset, data.ConcatDataset) \
                else trainloader.dataset.max_skip

            if (epoch + 1) % opt.epoch_per_test == 0:

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss': train_loss,
                    'minloss': minloss,
                    'optimizer': optimizer.state_dict(),
                    'max_skip': skips,
                }, epoch + 1, is_best, checkpoint=opt.checkpoint_STM, filename=opt.mode)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': att.state_dict(),
                    'loss': train_loss,
                    'minloss': minloss,
                    'optimizer': optimizer.state_dict(),
                    'max_skip': skips,
                }, epoch + 1, is_best, checkpoint=opt.checkpoint_att, filename=opt.mode)

        logger.close()

        print('minimum loss:')
        print(minloss)




def train(trainloader, model, Att_model,  criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold):
    # switch to train mode
    with torch.cuda.device(1):
        data_time = AverageMeter()
        loss = AverageMeter()

        end = time.time()

        bar = Bar('Processing', max=len(trainloader))
        optimizer.zero_grad()

        for batch_idx, data in enumerate(trainloader):
            frames, masks, objs, infos = data
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                frames = frames.cuda()
                masks = masks.cuda()
                objs = objs.cuda()
                # em = torch.zeros(1, objs + 1, H, W).cuda()

            objs[objs == 0] = 1

            N, T, C, H, W = frames.size()
            total_loss = 0.0
            for idx in range(N):
                frame = frames[idx]
                mask = masks[idx]
                num_objects = objs[idx]
                keys = []
                vals = []

                keys3 = []
                vals3 = []

                #pre-save
                for t in range(opt.save_freq):
                    key, val, _, key3, val3, _ = model(frame=frame[t:t + 1], mask=mask[t:t+1],num_objects=num_objects)
                    keys.append(key)
                    vals.append(val)

                    keys3.append(key3)
                    vals3.append(val3)

                # attention segment memory
                for t in range(opt.save_freq, T):
                    tmp_key_local = torch.stack(keys[-opt.save_freq:])
                    tmp_val_local = torch.stack(vals[-opt.save_freq:])

                    tmp_key_local3 = torch.stack(keys3[-opt.save_freq:])
                    tmp_val_local3 = torch.stack(vals3[-opt.save_freq:])

                    shuffle_keys = keys.copy()
                    shuffle_vals = vals.copy()
                    shuffle(shuffle_keys)
                    shuffle(shuffle_vals)
                    tmp_key_global = torch.stack(shuffle_keys[-opt.save_freq:])
                    tmp_val_global = torch.stack(shuffle_vals[-opt.save_freq:])

                    shuffle_keys3 = keys3.copy()
                    shuffle_vals3 = vals3.copy()
                    shuffle(shuffle_keys3)
                    shuffle(shuffle_vals3)
                    tmp_key_global3 = torch.stack(shuffle_keys3[-opt.save_freq:])
                    tmp_val_global3 = torch.stack(shuffle_vals3[-opt.save_freq:])

                    #attention
                    tmp_key_local = Att_model(f=tmp_key_local,tag='att_in_local')
                    tmp_val_local = Att_model(f=tmp_val_local,tag='att_out_local')
                    tmp_key_global = Att_model(f=tmp_key_global,tag='att_in_global')
                    tmp_val_global = Att_model(f=tmp_val_global,tag='att_out_global')

                    tmp_key_local3 = Att_model(f=tmp_key_local3,tag='att_in_local3')
                    tmp_val_local3 = Att_model(f=tmp_val_local3,tag='att_out_local3')
                    tmp_key_global3 = Att_model(f=tmp_key_global3,tag='att_in_global3')
                    tmp_val_global3 = Att_model(f=tmp_val_global3,tag='att_out_global3')

                    tmp_key = tmp_key_local + tmp_key_global
                    tmp_val = tmp_val_local + tmp_val_global

                    tmp_key3 = tmp_key_local3 + tmp_key_global3
                    tmp_val3 = tmp_val_local3 + tmp_val_global3

                    #segment
                    logits, ps, logits_simple = model(frame=frame[t:t + 1],keys=tmp_key, values=tmp_val, keys3=tmp_key3, values3=tmp_val3, num_objects=num_objects)
                    out = torch.softmax(logits, dim=1)
                    out_simple = torch.softmax(logits_simple, dim=1)

                    # memorize
                    with torch.no_grad():
                        key, val, _ , key3, val3, _= model(frame=frame[t:t + 1], mask=out, num_objects=num_objects)
                        keys.append(key)
                        vals.append(val)

                        keys3.append(key3)
                        vals3.append(val3)

                        if t > opt.save_freq_max:
                            keys.pop(0)
                            vals.pop(0)
                            keys3.pop(0)
                            vals3.pop(0)

                    gt = mask[t:t + 1]
                    total_loss = total_loss + criterion(out, gt, num_objects)
            total_loss = total_loss / (N * (T-opt.save_freq))

            # record loss
            if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:
                loss.update(total_loss.item(), 1)

            # compute gradient and do SGD step (divided by accumulated steps)
            total_loss /= iter_size

            total_loss.backward()

            if (batch_idx + 1) % iter_size == 0:
                optimizer.step()
                model.zero_grad()
                Att_model.zero_grad()

            # measure elapsed time
            end = time.time()
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}'.format(
                batch=batch_idx + 1,
                size=len(trainloader),
                data=data_time.val,
                loss=loss.avg
            )
            print('-'*20 + str(loss.avg))
            bar.next()
        bar.finish()

        return loss.avg



if __name__ == '__main__':
    main()
