from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate, mask_iou
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
from progress.bar import Bar
from collections import OrderedDict

from options import OPTION as opt
from matplotlib import pyplot as plt
import time
from random import shuffle
MAX_FLT = 1e6

# Use CUDA
device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

def main():
    # Data
    print('==> Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)

    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
    )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)
    # Model
    print("==> creating model")

    net = STM(opt.keydim, opt.valdim)
    att = Att(save_freq=opt.save_freq, keydim=opt.keydim, valdim=opt.valdim)

    print('    Total params: %.2fM' % ((sum(p.numel() for p in net.parameters())
    + sum(p.numel() for p in att.parameters())) / 1000000.0))

    # set eval to freeze batchnorm update
    net.eval()
    att.eval()

    with torch.cuda.device(1):
        if use_gpu:
            net.to(device)

        if use_gpu:
            att = att.cuda()

    # set training parameters
    for p in net.parameters():
        p.requires_grad = False
    for p in att.parameters():
        p.requires_grad = False

    # Resume
    title = 'STM'

    if opt.resume_STM:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume_STM))
        assert os.path.isfile(opt.resume_STM), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_STM, map_location=device)
        net.load_state_dict(checkpoint['state_dict'],strict=False)

    if opt.resume_ATT:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume_ATT))
        assert os.path.isfile(opt.resume_ATT), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume_ATT)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        att.load_state_dict(checkpoint['state_dict'],strict=False)

    # Train and val
    print('==> Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))

    test(testloader,
         model=net,
         Att_model=att,
         use_cuda=use_gpu,
         opt=opt)

    print('==> Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))


def test(testloader, model, Att_model, use_cuda, opt):
    time_cost = []

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                with torch.cuda.device(1):
                    frames = frames.to(device)
                    # em = torch.zeros(1, objs + 1, H, W).to(ps.device)

            frames = frames[0]
            num_objects = objs[0]
            info = infos[0]


            T, _, H, W = frames.shape
            pred = []
            keys = []
            vals = []

            keys3 = []
            vals3 = []


            # inintial for pre-frames
            for t in range(opt.save_freq):
                # print('This is frame----',str(t))
                logits_simple = model(frame=frames[t:t+1, :, :, :])
                out = torch.softmax(logits_simple, dim=1)
                pred.append(out)

                # memorize
                key, val, _, key3, val3, _ = model(frame=frames[t:t+1, :, :, :], mask=out, num_objects=num_objects)

                keys.append(key)
                vals.append(val)

                keys3.append(key3)
                vals3.append(val3)

            start = time.time()

            for t in range(opt.save_freq, T): #T
                # print('This is frame----',str(t))

                # segment
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


                logits, ps, _ = model(frame=frames[t:t + 1, :, :, :], keys=tmp_key, values=tmp_val, keys3=tmp_key3, values3=tmp_val3,
                                   num_objects=num_objects)

                out = torch.softmax(logits, dim=1)
                pred.append(out)

                # memorize
                key, val, _ , key3, val3, _= model(frame=frames[t:t+1, :, :, :], mask=out,  num_objects=num_objects)

                keys.append(key)
                vals.append(val)

                keys3.append(key3)
                vals3.append(val3)

                if t > opt.save_freq_max:
                    keys.pop(0)
                    vals.pop(0)

                    keys3.pop(0)
                    vals3.pop(0)

            end = time.time()
            frames_num = T - opt.save_freq
            tmp_time = end - start
            time_cost.append(tmp_time)
            print(info['name']+' frames_num: ' + str(frames_num) + ' Time cost: ' + str(tmp_time))
            print('testing fps: ' + str(1 / (tmp_time / frames_num)))


            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            write_mask(pred, info, opt, directory=opt.output_dir)

        time_sum = 0
        for _, val in enumerate (time_cost):
            time_sum += val / (frames_num)
        print('average fps: ' + str(1 / (time_sum / len(time_cost))))

        return


if __name__ == '__main__':
    main()
