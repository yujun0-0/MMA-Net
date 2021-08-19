import torch
import torch.nn as nn



class Att_in(nn.Module):
    def __init__(self,save_freq, keyDim ):
        super(Att_in, self).__init__()
        self.conv1_1 = nn.Conv2d(save_freq*keyDim,keyDim,kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(keyDim, keyDim, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(keyDim, keyDim, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = nn.Conv2d(keyDim, save_freq, kernel_size=1, bias=False)
    def forward(self, f):
        _, b, C, H, W = f.size()
        f_att= f.permute(1, 0, 2, 3, 4).contiguous().view(b,-1,H,W)
        f_att = self.conv1_1(f_att)
        f_att = self.conv3_1(f_att)
        f_att = self.conv3_2(f_att)
        f_att = self.conv1_2(f_att)
        f_att = torch.softmax(f_att, dim=1)
        f = torch.unsqueeze(f_att, dim=2) * f.permute(1, 0, 2, 3, 4)
        f = torch.sum(f, dim=1).view(b,-1,H*W).permute(0,2,1)
        return f

class Att_out(nn.Module):
    def __init__(self, save_freq, valdim):
        super(Att_out, self).__init__()
        self.conv1_1 = nn.Conv2d(save_freq*valdim,valdim,kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(valdim, valdim, kernel_size=3, padding=1, stride=1)
        self.conv3_2 = nn.Conv2d(valdim, valdim, kernel_size=3, padding=1, stride=1)
        self.conv1_2 = nn.Conv2d(valdim, save_freq, kernel_size=1, bias=False)
    def forward(self, f):
        _, b, C, H, W = f.size()
        f_att= f.permute(1, 0, 2, 3, 4).contiguous().view(b,-1,H,W)
        f_att = self.conv1_1(f_att)
        f_att = self.conv3_1(f_att)
        f_att = self.conv3_2(f_att)
        f_att = self.conv1_2(f_att)
        f_att = torch.softmax(f_att, dim=1)
        f = torch.unsqueeze(f_att, dim=2) * f.permute(1, 0, 2, 3, 4)
        f = torch.sum(f, dim=1).view(b,-1,H*W).permute(0,2,1)
        return f

class Att(nn.Module):
    def __init__(self,save_freq, keydim, valdim):
        super(Att,self).__init__()
        self.Att_in_local = Att_in(save_freq, keydim)
        self.Att_in_global = Att_in(save_freq, keydim)
        self.Att_out_local = Att_out(save_freq, valdim)
        self.Att_out_global = Att_out(save_freq, valdim)

        self.Att_in_local3 = Att_in(save_freq, keydim//2)
        self.Att_in_global3 = Att_in(save_freq, keydim//2)
        self.Att_out_local3 = Att_out(save_freq, valdim//2)
        self.Att_out_global3 = Att_out(save_freq, valdim//2)
    def forward(self, f, tag):
        if tag=='att_in_local':
            return  self.Att_in_local(f)
        if tag=='att_out_local':
            return  self.Att_out_local(f)
        if tag=='att_in_global':
            return  self.Att_in_global(f)
        if tag=='att_out_global':
            return  self.Att_out_global(f)

        if tag=='att_in_local3':
            return  self.Att_in_local3(f)
        if tag=='att_out_local3':
            return  self.Att_out_local3(f)
        if tag=='att_in_global3':
            return  self.Att_in_global3(f)
        if tag=='att_out_global3':
            return  self.Att_out_global3(f)