import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision import models
from .backbone import Encoder, Decoder, Bottleneck
import numpy as np

import time

from ..utils.utility import mask_iou


def Soft_aggregation(ps, max_obj):


    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W, device=ps.device)
    # em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))
    return logit


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)


    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_bg):

        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float()  # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024



        return r4, r3, r2, c1

class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        self.conv1x1 = nn.Conv2d(1024, 9, kernel_size=1, stride=1)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):


        f = in_f

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024

        r_pred = self.conv1x1(r4)



        return r4, r3, r2, c1, r_pred

class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, inplane, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)  # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2, f):


        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)  # out: 1/8, 256
        m2 = self.RF2(r2, m3)  # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False)



        return p

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w


        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape

        qi = q_in.view(-1, C, H * W)
        p = torch.bmm(m_in, qi)  # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1)  # no x centers x hw

        mo = m_out.permute(0, 2, 1)  # no x c x centers
        mem = torch.bmm(mo, p)  # no x c x hw
        mem = mem.view(no, vd, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)



        return mem_out, p

class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class STM(nn.Module):
    def __init__(self, keydim, valdim, phase='test', mode='recurrent', iou_threshold=0.5):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M()
        self.Encoder_Q = Encoder_Q()

        self.keydim = keydim
        self.valdim = valdim

        self.KV_M_r4 = KeyValue(1024, keydim=self.keydim, valdim=self.valdim)
        self.KV_Q_r4 = KeyValue(1024, keydim=self.keydim, valdim=self.valdim)

        self.KV_M_r3 = KeyValue(512, keydim=self.keydim//2, valdim=self.valdim//2)
        self.KV_Q_r3 = KeyValue(512, keydim=self.keydim//2, valdim=self.valdim//2)
        # self.Routine = DynamicRoutine(channel, iters, centers)

        self.Memory = Memory()
        self.Decoder = Decoder(2 * self.valdim, 256)
        self.phase = phase
        self.mode = mode
        self.iou_threshold = iou_threshold

        self.flag = 0
        self.count = 0

        assert self.phase in ['train', 'test']

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects):


        # memorize a frame
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        bg_batch = []
        try:
            for o in range(1, num_objects + 1):  # 1 - no
                frame_batch.append(frame)
                mask_batch.append(masks[:, o])

            for o in range(1, num_objects + 1):
                bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))

            # make Batch
            frame_batch = torch.cat(frame_batch, dim=0)
            mask_batch = torch.cat(mask_batch, dim=0)
            bg_batch = torch.cat(bg_batch, dim=0)
        except RuntimeError as re:
            print(re)
            print(num_objects)
            raise re

        from matplotlib import pyplot as plt
        r4, r3, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch)  # no, c, h, w

        _, c, h, w = r4.size()
        memfeat = r4
        # memfeat = self.Routine(memfeat, maskb)
        # memfeat = memfeat.view(-1, c)
        k4, v4 = self.KV_M_r4(memfeat)
        k3, v3 = self.KV_M_r3(r3)
        # k4 = k4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim)
        # v4 = v4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)



        return k4, v4, r4, k3, v3, r3

    def segment(self, frame, keys, values, keys3, values3, num_objects):

        # segment one input frame

        r4, r3, r2, _, r_pred = self.Encoder_Q(frame)
        pred_simple = F.interpolate(r_pred, size=frame.shape[2:], mode='bilinear', align_corners=False)
        pred_simple = torch.clamp(pred_simple, 1e-7, 1 - 1e-7)
        pred_simple = torch.log((pred_simple / (1 - pred_simple)))
        n, c, h, w = r4.size()
        # r4 = r4.permute(0, 2, 3, 1).contiguous().view(-1, c)
        k4, v4 = self.KV_Q_r4(r4)  # 1, dim, H/16, W/16
        k3, v3 = self.KV_Q_r3(r3)
        # k4 = k4.view(n, self.keydim, -1).permute(0, 2, 1)
        # v4 = v4.view(n, self.valdim, -1).permute(0, 2, 1)

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects, -1, -1, -1), v4.expand(num_objects, -1, -1, -1)
        # r3e, r2e = r3.expand(num_objects, -1, -1, -1), r2.expand(num_objects, -1, -1, -1)
        r2e = r2.expand(num_objects, -1, -1, -1)

        k3e, v3e = k3.expand(num_objects, -1, -1, -1), v3.expand(num_objects, -1, -1, -1)


        m4, _ = self.Memory(keys, values, k4e, v4e)
        m3, _ = self.Memory(keys3, values3, k3e, v3e)


        logit = self.Decoder(m4, m3, r2e, frame)
        ps = F.softmax(logit, dim=1)[:, 1]  # no, h, w
        # ps = indipendant possibility to belong to each object
        logit = Soft_aggregation(ps, num_objects)  # 1, K, H, W



        return logit, ps, pred_simple

    def segment_simple(self, frame):
        # segment one input frame
        s=time.time()

        r4, r3, r2, _, r_pred = self.Encoder_Q(frame)
        pred_simple = F.interpolate(r_pred, size=frame.shape[2:], mode='bilinear', align_corners=False)
        pred_simple = torch.clamp(pred_simple, 1e-7, 1 - 1e-7)
        pred_simple = torch.log((pred_simple / (1 - pred_simple)))

        e = time.time()
        # print('segment simple cost ms', str((e - s)*1000))

        return  pred_simple


    def forward(self, frame, mask=None, keys=None, values=None, keys3=None, values3=None, num_objects=None):

        if mask is not None:  # keys
            return self.memorize(frame, mask, num_objects)
        elif num_objects is None:
            return self.segment_simple(frame)
        else:
            return self.segment(frame, keys, values, keys3, values3, num_objects)
