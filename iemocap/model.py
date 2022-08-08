import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import torch.nn as nn
import argparse
import os

import warnings
warnings.filterwarnings(action='ignore')

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle
import csv
import math

import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)  # batch*seq_len, 1
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i].tolist()) if i < 4 else pad_sequence(dat[i].tolist(), True) if i < 6 else dat[
            i].tolist() for i in dat]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, \
        self.trainVid, self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'user' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, n_classes, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, _ = pickle.load(open(path, 'rb'))
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i].tolist()) if i < 3 else pad_sequence(dat[i].tolist(), True) if i < 5 else dat[
            i].tolist() for i in dat]


class DailyDialogueDataset(Dataset):

    def __init__(self, split, path):

        self.Speakers, self.InputSequence, self.InputMaxSequenceLength, \
        self.ActLabels, self.EmotionLabels, self.trainId, self.testId, self.validId = pickle.load(open(path, 'rb'))

        if split == 'train':
            self.keys = [x for x in self.trainId]
        elif split == 'test':
            self.keys = [x for x in self.testId]
        elif split == 'valid':
            self.keys = [x for x in self.validId]

        self.len = len(self.keys)

    def __getitem__(self, index):
        conv = self.keys[index]

        return torch.LongTensor(self.InputSequence[conv]), \
               torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.Speakers[conv]]), \
               torch.FloatTensor([1] * len(self.ActLabels[conv])), \
               torch.LongTensor(self.ActLabels[conv]), \
               torch.LongTensor(self.EmotionLabels[conv]), \
               self.InputMaxSequenceLength[conv], \
               conv

    def __len__(self):
        return self.len


class DailyDialoguePadCollate:

    def __init__(self, dim=0):
        self.dim = dim

    def pad_tensor(self, vec, pad, dim):
        pad_size = list(vec.shape)
        pad_size[dim] = pad - vec.size(dim)
        return torch.cat([vec, torch.zeros(*pad_size).type(torch.LongTensor)], dim=dim)

    def pad_collate(self, batch):
        # find longest sequence
        max_len = max(map(lambda x: x.shape[self.dim], batch))

        # pad according to max_len
        batch = [self.pad_tensor(x, pad=max_len, dim=self.dim) for x in batch]

        # stack all
        return torch.stack(batch, dim=0)

    def __call__(self, batch):
        dat = pd.DataFrame(batch)

        return [self.pad_collate(dat[i]).transpose(1, 0).contiguous() if i == 0 else \
                    pad_sequence(dat[i]) if i == 1 else \
                        pad_sequence(dat[i], True) if i < 5 else \
                            dat[i].tolist() for i in dat]


from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    fs=[]
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        # [L,B,D],[L,B,D],[L,B,D],[L,B,I],[B,L],[B,L]
        textf, visuf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        # Subsequence mask.
        submask = (1 - torch.triu(torch.ones((1, umask.size(1), umask.size(1)), device=umask.device), diagonal=1)).bool()
        # Padding mask.
        padmask = umask.unsqueeze(-2).bool()
        # Self(Speaker2Speaker) mask.
        smask = (qmask[:,:,0]*2+qmask[:,:,1]).transpose(0,1) # [B,L]
        smask = smask.unsqueeze(1).expand(-1, smask.size(1), -1) # [B,L,L]
        smask = (smask.transpose(1,2) == smask).bool() # [B,L,L]
        # Cross(Speaker2Listener) mask.
        cmask = (qmask[:,:,0]*2+qmask[:,:,1]).transpose(0,1) # [B,L]
        cmask = cmask.unsqueeze(1).expand(-1, cmask.size(1), -1) # [B,L,L]
        cmask = (cmask.transpose(1,2) == cmask).bool() # [B,L,L]
        # Final mask.
        smask = submask & padmask & smask
        cmask = submask & padmask & cmask
        smask = smask.cuda() if cuda else smask
        cmask = cmask.cuda() if cuda else cmask
        # Query weight.
        weight = torch.zeros(1, umask.size(1), umask.size(1))
        for i in range(weight.size(1)):
            for j in range(i+1):
                weight[0,i,j] = math.exp(1*(j-i))

        # log_prob, _ = model(torch.cat((textf, acouf, visuf), dim=-1).transpose(0, 1), smask, cmask)  # batch, seq_len, n_classes
        log_prob, _ ,f= model(textf.transpose(0, 1), acouf.transpose(0, 1), visuf.transpose(0, 1), smask, cmask)  # batch, seq_len, n_classes
        lp_ = log_prob.view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        fs.append(f.data.cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            vids += data[-1]

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
        fs = np.concatenate(fs,axis=0)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [],[]

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids],fs


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [b x n x lq x d_k], [b x n x lq x d_k], [b x n x lq x d_v]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, mask


class DropScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d0, d1, temperature, attn_dropout=0.1, args=None):
        super().__init__()
        self.args = args
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.dc = nn.Sequential(nn.Linear(d0 + d1, d0 + d1), nn.ReLU(),
                                nn.Linear(d0 + d1, 2))
        # self.dc = nn.Linear(d0 + d1, 2)

    def gumbel_softmax(self, logits, tau, hard: bool = False, eps: float = 1e-10, dim: int = -1, training=False):
        # logits = F.softmax(logits,dim=-1)
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        if training:
            gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        else:
            # print(logits[0][0][0],gumbels[0][0][0])
            gumbels = (logits) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, q, k, v, d0, d1, mask=None, pre_decisions=None):
        """
        q, k, v, d0: [b x n x lq x d_k], [b x n x lq x d_k], [b x n x lq x d_v], [b x lq x d_d]
        """
        b, n, lq, dk, dv, dd = q.size(0), q.size(1), q.size(2), q.size(3), v.size(3), d0.size(2)
        # f = torch.cat(
        #     [q.repeat(1, 1, 1, lq).view(b, n, lq, lq, dk).permute(0,2,3,1,4).contiguous().view(b,1,lq,lq,n*dk),
        #      k.repeat(1, 1, lq, 1).view(b, n, lq, lq, dk).permute(0,2,3,1,4).contiguous().view(b,1,lq,lq,n*dk)], dim=4)
        f = torch.cat(
            [d0.repeat(1, 1, lq).view(b, lq, lq, dd),
             d1.repeat(1, lq, 1).view(b, lq, lq, dd)], dim=3)
        f = self.dc(f)
        distributions = self.gumbel_softmax(f, tau=1, hard=True, dim=-1, training=self.training).unsqueeze(
            1)  # [b x 1 x lq x lq x 2]

        decisions = distributions[:, :, :, :, -1].bool()  # [b x 1 x lq x lq]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            decisions = mask & decisions
            attn = attn.masked_fill(decisions == 0, -1e9)
        else:
            attn = attn.masked_fill(decisions == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, decisions


class GlobalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, mask=None):
        """
        q, k, v: [b x n x lq x d_k], [b x n x lq x d_k]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        return attn


class DropGlobalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d0, d1, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.dc = nn.Sequential(nn.Linear(d0 + d1, d0 + d1), nn.ReLU(),
                                nn.Linear(d0 + d1, 2))

    def gumbel_softmax(self, logits, tau, hard: bool = False, eps: float = 1e-10, dim: int = -1, training=False):
        logits = F.softmax(logits, dim=-1)
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        if training:
            gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        else:
            gumbels = (logits) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def forward(self, q, k, d0, d1, mask=None):
        """
        q, k, v: [b x n x lq x d_k], [b x n x lq x d_k]
        """
        b, n, lq, dk, dd = q.size(0), q.size(1), q.size(2), q.size(3), d0.size(2)
        f = torch.cat(
            [d1.repeat(1, 1, lq).view(b, lq, lq, dd),
             d0.repeat(1, lq, 1).view(b, lq, lq, dd)], dim=3)
        f = self.dc(f)
        distributions = self.gumbel_softmax(f, tau=1, hard=True, dim=-1, training=self.training).unsqueeze(
            1)  # [b x 1 x lq x lq x 2]
        decisions = distributions[:, :, :, :, -1] * -1e9  # [b x 1 x lq x lq]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn + decisions
            attn = attn.masked_fill(mask == 0, -1e9)
        else:
            attn = attn + decisions

        return attn


class CrossGlobalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, mask=None):
        """
        q, k, v: [b x n x lq x d_k], [b x n x lq x d_k]
        """
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        return attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, selector=False, args=None):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.selector = selector
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        if selector:
            self.w_ds0 = nn.Linear(d_model, n_head * d_v, bias=False)
            self.w_ds1 = nn.Linear(d_model, n_head * d_v, bias=False)
            self.attention = DropScaledDotProductAttention(d0=n_head * d_v, d1=n_head * d_v, temperature=d_k ** 0.5,
                                                           args=args)
        else:
            self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, decisions=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        if self.selector:
            d0 = self.w_ds0(q).view(sz_b, len_q, n_head * d_v)
            d1 = self.w_ds1(v).view(sz_b, len_v, n_head * d_v)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        if self.selector:
            q, attn, decisions = self.attention(q, k, v, d0, d1, mask=mask, pre_decisions=decisions)
        else:
            q, attn, decisions = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn, decisions


class GlobalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, selector=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.selector = selector
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        if selector:
            self.w_ds0 = nn.Linear(d_model, n_head * d_v, bias=False)
            self.w_ds1 = nn.Linear(d_model, n_head * d_v, bias=False)
            self.attention = DropGlobalScaledDotProductAttention(n_head * d_v, n_head * d_v, temperature=d_k ** 0.5)
        else:
            self.attention = GlobalScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_t, input_a, input_v=None, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        lis = [input_t, input_a, input_v] if input_v is not None else [input_t, input_a]
        qs = []
        for i in range(len(lis)):
            q = lis[i]
            residual = q
            sz_b, len_q = q.size(0), q.size(1)
            if self.selector:
                d0 = self.w_ds0(q).view(sz_b, len_q, n_head * d_v)
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            q = q.transpose(1, 2)
            attns = []
            vs = []
            for j in range(len(lis)):
                k = lis[j]
                v = k  # [B, L, D]
                len_k, len_v = k.size(1), v.size(1)
                # Pass through the pre-attention projection: b x lq x (n*dv)
                # Separate different heads: b x lq x n x dv
                k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
                if self.selector:
                    d1 = self.w_ds1(v).view(sz_b, len_v, n_head * d_v)
                v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

                # Transpose for attention dot product: b x n x lq x dv
                k, v = k.transpose(1, 2), v.transpose(1, 2)
                vs.append(v.unsqueeze(2))
                if self.selector:
                    attn = self.attention(q, k, d0, d1, mask=mask).unsqueeze(3)  # [B,N,L,1,L]
                    attns.append(attn)
                else:
                    attn = self.attention(q, k, mask=mask).unsqueeze(3)  # [B,N,L,1,L]
                    attns.append(attn)
            attns = torch.cat(attns, dim=3)  # [B,N,L,3,L]
            vs = torch.cat(vs, dim=2)  # [B,N,3,L,D]
            attns = self.dropout(F.softmax(attns.view(sz_b, n_head, len_q, -1), dim=-1))
            q = torch.matmul(attns, vs.view(sz_b, n_head, len(lis) * len_q, -1))  # [B,N,L,D]
            # Transpose to move the head dimension back: b x lq x n x dv
            # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
            q = self.dropout(self.fc(q))
            q += residual
            q = self.layer_norm(q)
            qs.append(q)
        if input_v is None:
            qs.append(None)

        return qs, None


class PGlobalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = GlobalScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_t, input_a, input_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        lis = [input_t, input_a, input_v]
        qs = []
        for i in range(len(lis)):
            q = lis[i]
            residual = q
            sz_b, len_q = q.size(0), q.size(1)
            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            q = q.transpose(1, 2)
            attns = []
            vs = []
            for j in range(len(lis)):
                if j == i:
                    continue
                k = lis[j]
                v = k  # [B, L, D]
                len_k, len_v = k.size(1), v.size(1)
                # Pass through the pre-attention projection: b x lq x (n*dv)
                # Separate different heads: b x lq x n x dv
                k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
                v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

                # Transpose for attention dot product: b x n x lq x dv
                k, v = k.transpose(1, 2), v.transpose(1, 2)
                vs.append(v.unsqueeze(2))

                attn = self.attention(q, k, mask=mask).unsqueeze(3)  # [B,N,L,1,L]
                attns.append(attn)
            attns = torch.cat(attns, dim=3)  # [B,N,L,3,L]
            vs = torch.cat(vs, dim=2)  # [B,N,3,L,D]
            attns = self.dropout(F.softmax(attns.view(sz_b, n_head, len_q, -1), dim=-1))
            q = torch.matmul(attns, vs.view(sz_b, n_head, 2 * len_q, -1))  # [B,N,L,D]
            # Transpose to move the head dimension back: b x lq x n x dv
            # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
            q = self.dropout(self.fc(q))
            q += residual
            q = self.layer_norm(q)
            qs.append(q)

        return qs, None


class CrossGlobalMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.ModuleList([nn.Linear(d_model, n_head * d_k, bias=False),
                                   nn.Linear(d_model, n_head * d_k, bias=False),
                                   nn.Linear(d_model, n_head * d_k, bias=False)])
        self.w_ks = nn.ModuleList([nn.Linear(d_model, n_head * d_k, bias=False),
                                   nn.Linear(d_model, n_head * d_k, bias=False),
                                   nn.Linear(d_model, n_head * d_k, bias=False)])
        self.w_vs = nn.ModuleList([nn.Linear(d_model, n_head * d_v, bias=False),
                                   nn.Linear(d_model, n_head * d_v, bias=False),
                                   nn.Linear(d_model, n_head * d_v, bias=False)])
        self.fc = nn.ModuleList([nn.Linear(n_head * d_v, d_model, bias=False),
                                 nn.Linear(n_head * d_v, d_model, bias=False),
                                 nn.Linear(n_head * d_v, d_model, bias=False)])

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6),
                                         nn.LayerNorm(d_model, eps=1e-6),
                                         nn.LayerNorm(d_model, eps=1e-6)])
        self.readout = nn.ModuleList([MultiHeadReadOutAttention(1, d_model, 2),
                                      MultiHeadReadOutAttention(1, d_model, 2),
                                      MultiHeadReadOutAttention(1, d_model, 2)])

    def forward(self, input_t, input_a, input_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        lis = [input_t, input_a, input_v]  # [B, L, D]
        res = []
        for i in range(len(lis)):
            k = v = lis[i]
            sz_b, len_k, len_v = k.size(0), k.size(1), v.size(1)
            k = self.w_ks[0](k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs[0](v).view(sz_b, len_v, n_head, d_v)
            # Transpose for attention dot product: b x n x lq x dv
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            qs = []
            for j in range(len(lis)):
                if j == i:
                    continue
                q = lis[j]  # [B, L, D]
                sz_b, len_q = q.size(0), q.size(1)
                # Pass through the pre-attention projection: b x lq x (n*dv)
                # Separate different heads: b x lq x n x dv
                q = self.w_qs[j](q).view(sz_b, len_q, n_head, d_k)
                q = q.transpose(1, 2)

                q, attn = self.attention(q, k, v, mask=mask)  # [B,N,L,dv]

                # Transpose to move the head dimension back: b x lq x n x dv
                # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
                q = q.transpose(1, 2).contiguous().view(sz_b, len_q, 1, -1)
                q = self.dropout(self.fc[0](q))
                q = self.layer_norm[0](q)
                qs.append(q)
            v = self.readout[0](torch.cat(qs, dim=2))[0].squeeze(2)
            res.append(v)
        return res, None


class MultiHeadReadOutAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, l, d_inner=2048, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.empty((1, 1, n_head, l, d_model), requires_grad=True))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.temperature = self.d_model ** 0.5
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = PositionwiseFeedForward(d_model, d_inner)
        # self.fc_residual = nn.Linear(l_q, n_head, bias=False)

    def forward(self, x, mask=None):
        # x: [B, L, M, D], mask: [B, L, M]
        # residual = self.dropout(self.fc_residual(x.transpose(1,2)).transpose(1,2))
        x = x.unsqueeze(-3)  # [B, L, 1, M, D]
        v = self.v(x)  # [B, L, 1, M, D]
        #         v = x  # [B, L, 1, M, D]
        attn = torch.mul(x / self.temperature, self.w)  # [B, L, N, M, D]
        attn = torch.sum(attn, dim=-1, keepdim=False)  # [B, L, N, M]
        if mask is not None:
            mask = mask.unsqueeze(-2)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(v.transpose(-1, -2), attn.unsqueeze(-1)).squeeze(-1)  # [B, L, N, D]
        output = self.dropout(output)
        # output += residual
        output = self.layer_norm(output)
        output = self.ffn(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, selector=False, args=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, selector=selector, args=args)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, decisions=None):
        enc_output, enc_slf_attn, decisions = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, decisions=decisions)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn, decisions


class GlobalEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, selector=False):
        super(GlobalEncoderLayer, self).__init__()
        self.glb_attn = GlobalMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, selector=selector)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input_t, input_a, input_v=None, slf_attn_mask=None):
        [input_t, input_a, input_v], enc_slf_attn = self.glb_attn(
            input_t, input_a, input_v, mask=slf_attn_mask)
        input_t = self.pos_ffn(input_t)
        input_a = self.pos_ffn(input_a)
        if input_v is not None:
            input_v = self.pos_ffn(input_v)
        return input_t, input_a, input_v, enc_slf_attn


class CrossGlobalEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(CrossGlobalEncoderLayer, self).__init__()
        self.glb_attn = CrossGlobalMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.pos_ffn3 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, input_t, input_a, input_v, slf_attn_mask=None):
        [input_t, input_a, input_v], enc_slf_attn = self.glb_attn(
            input_t, input_a, input_v, mask=slf_attn_mask)
        input_t = self.pos_ffn1(input_t)
        input_a = self.pos_ffn1(input_a)
        input_v = self.pos_ffn1(input_v)
        return input_t, input_a, input_v, enc_slf_attn


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class SpeakerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_classes, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, selector=False, args=None):

        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, selector, args=args) if i == 0 else
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, False, args=args)
            for i in range(n_layers)])

    def forward(self, enc_output, src_mask=None, return_attns=False, decisions=None):

        enc_slf_attn_list = []
        decision_list = []

        # -- Forward
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn, decisions = enc_layer(enc_output, slf_attn_mask=src_mask, decisions=None)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
            decision_list += [decisions] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list, decision_list
        return enc_output, None, None


class ModalityEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_classes, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_output, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class GlobalEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_classes, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, selector=False):

        super().__init__()
        self.layer_stack = nn.ModuleList([
            GlobalEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, selector=selector) if i == 0 else
            GlobalEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, selector=False)
            for i in range(n_layers)])

    def forward(self, output_t, output_a, output_v=None, src_mask=None, return_attns=False):
        # t_decisions: [B,1,L,L]
        enc_slf_attn_list = []

        # -- Forward
        for enc_layer in self.layer_stack:
            output_t, output_a, output_v, enc_slf_attn = enc_layer(output_t, output_a, output_v, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return output_t, output_a, output_v, enc_slf_attn_list
        return output_t, output_a, output_v


class CrossGlobalEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_classes, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        self.layer_stack = nn.ModuleList([
            CrossGlobalEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, output_t, output_a, output_v, src_mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        for enc_layer in self.layer_stack:
            output_t, output_a, output_v, enc_slf_attn = enc_layer(output_t, output_a, output_v, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return output_t, output_a, output_v, enc_slf_attn_list
        return output_t, output_a, output_v


class MMModel(nn.Module):
    ''' Encoding for each modality. '''

    def __init__(
            self, n_classes, dt, da, dv, d_model, n_layers, n_head, d_k, d_v,
            d_inner, dropout=0.1, n_position=200, scale_emb=False, selector=False, args=None):
        super().__init__()
        self.args = args
        self.t = nn.Linear(dt, d_model)
        self.a = nn.Linear(da, d_model)
        self.v = nn.Linear(dv, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model
        self.scale_emb = scale_emb
        self.selector = selector
        self.smax_fc = nn.Linear(d_model, n_classes)
        self.norm3a = nn.BatchNorm1d(1024, affine=True)
        self.norm3b = nn.BatchNorm1d(1024, affine=True)
        self.norm3c = nn.BatchNorm1d(1024, affine=True)
        self.norm3d = nn.BatchNorm1d(1024, affine=True)

        self.se_t = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)
        self.se_a = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)
        self.se_v = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)
        self.ce_t = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)
        self.ce_a = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)
        self.ce_v = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                   self.selector, args)

        # self.sme = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        # self.cme = SpeakerEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)

        # self.gse = CrossGlobalEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        # self.gce = CrossGlobalEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.ge = GlobalEncoder(n_classes, d_model, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout,
                                False)
        # self.readout_t = MultiHeadReadOutAttention(1, d_model, 2)
        # self.readout_a = MultiHeadReadOutAttention(1, d_model, 2)
        # self.readout_v = MultiHeadReadOutAttention(1, d_model, 2)
        # self.readout_ta = MultiHeadReadOutAttention(1, d_model, 2)
        # self.readout_av = MultiHeadReadOutAttention(1, d_model, 2)
        # self.readout_tv = MultiHeadReadOutAttention(1, d_model, 2)
        self.readout_tav = MultiHeadReadOutAttention(1, d_model, 3)
        self.readout_ta = MultiHeadReadOutAttention(1, d_model, 2)

    def forward(self, tf, af, vf=None, self_mask=None, cross_mask=None, return_attns=False):
        enc_slf_attn_list = []

        r1, r2, r3, r4 = tf[:, :, 0, :], tf[:, :, 1, :], tf[:, :, 2, :], tf[:, :, 3, :]  # [L,B,D]
        lq, dm = r1.size(1), r1.size(2)
        r1 = self.norm3a(r1.reshape(-1, dm)).reshape(-1, lq, dm)
        r2 = self.norm3b(r2.reshape(-1, dm)).reshape(-1, lq, dm)
        r3 = self.norm3c(r3.reshape(-1, dm)).reshape(-1, lq, dm)
        r4 = self.norm3d(r4.reshape(-1, dm)).reshape(-1, lq, dm)
        tf = (r1 + r2 + r3 + r4) / 4

        # af=tf
        # -- Forward
        # if self.scale_emb:
        #     enc_output *= self.d_model ** 0.5
        tf = self.layer_norm(self.dropout(self.position_enc(self.t(tf))))
        af = self.layer_norm(self.dropout(self.position_enc(self.a(af))))
        if vf is not None:
            vf = self.layer_norm(self.dropout(self.position_enc(self.v(vf))))

        # Self-Speaker Self-Modal Encoding
        tf, _, _ = self.se_t(tf, self_mask)  # [B, L, D]
        af, _, _ = self.se_a(af, self_mask)  # [B, L, D]
        if vf is not None:
            vf, _, _ = self.se_v(vf, self_mask)  # [B, L, D]

        # Cross-Speaker Self-Modal Encoding
        tf, _, _ = self.ce_t(tf, cross_mask)  # [B, L, D]
        af, _, _ = self.ce_a(af, cross_mask)  # [B, L, D]
        if vf is not None:
            vf, _, _ = self.ce_v(vf, cross_mask)  # [B, L, D]


        # Self-Speaker Cross-Modal Encoding
        tf, af, vf = self.ge(tf, af, vf)

        # # Speaker Fusion
        # tf = torch.cat([tf1.unsqueeze(2), tf2.unsqueeze(2)], dim=2) # [B,L,2,D]
        # af = torch.cat([af1.unsqueeze(2), af2.unsqueeze(2)], dim=2)
        # if vf is not None:
        #     vf = torch.cat([vf1.unsqueeze(2), vf2.unsqueeze(2)], dim=2)
        # tf = self.readout_t(tf)[0].squeeze(-2)
        # af = self.readout_a(af)[0].squeeze(-2)
        # if vf is not None:
        #     vf = self.readout_v(vf)[0].squeeze(-2)

        # # Cross-Speaker Cross-Modal Encoding
        # tf, af, vf = self.gce(tf, af, vf, cross_mask) # [B,L,D]

        # # Modality Encoding
        # f = torch.cat([tf1.unsqueeze(2), af1.unsqueeze(2), vf1.unsqueeze(2)], dim=2) \
        #     .reshape(tf1.size(0) * tf1.size(1), 3, tf1.size(2))  # [B*L, 3, D]
        # f = self.sme(f).reshape(tf1.size(0), tf1.size(1), 3, tf1.size(2))  # [B, L, 3, D]
        # tf1 = f[:, :, 0, :].squeeze(2)
        # af1 = f[:, :, 1, :].squeeze(2)
        # vf1 = f[:, :, 2, :].squeeze(2)

        #         # Modality Encoding
        #         f = torch.cat([tf.unsqueeze(2), af.unsqueeze(2), vf.unsqueeze(2)], dim=2) \
        #             .reshape(tf.size(0) * tf.size(1), 3, tf.size(2))  # [B*L, 3, D]

        #         f = self.cme(f).reshape(tf.size(0), tf.size(1), 3, tf.size(2))  # [B, L, 3, D]
        #         tf = f[:, :, 0, :].squeeze(2)
        #         af = f[:, :, 1, :].squeeze(2)
        #         vf = f[:, :, 2, :].squeeze(2)

        #         Self-Speaker Cross-Modal Encoding
        #                 tf, af, vf = self.gse(tf, af, vf)

        # Modality Fusion
        #         ta = self.readout_ta(torch.cat([tf.unsqueeze(2), af.unsqueeze(2)], dim=2))[0] # [B,L,1,D]
        #         av = self.readout_av(torch.cat([af.unsqueeze(2), vf.unsqueeze(2)], dim=2))[0]
        #         tv = self.readout_tv(torch.cat([tf.unsqueeze(2), vf.unsqueeze(2)], dim=2))[0]
        #         f = torch.cat([ta, av, tv], dim=2)  # [B,L,3,D]
        if vf is not None:
            f = torch.cat([tf.unsqueeze(2), af.unsqueeze(2), vf.unsqueeze(2)], dim=2)  # [B,L,3,D]
            enc_output = self.readout_tav(f)[0].squeeze(-2)  # [B,L,D]
            # enc_output = torch.cat([tf, af, vf], dim=2)  # [B,L,3D]
            log_prob = F.log_softmax(self.smax_fc(enc_output), 2)
            # if return_attns:
            #     return log_prob, enc_output, enc_slf_attn_list
            return log_prob, enc_output, enc_output.reshape(enc_output.size(0) * enc_output.size(1), -1)
        else:
            f = torch.cat([tf.unsqueeze(2), af.unsqueeze(2)], dim=2)  # [B,L,2,D]
            enc_output = self.readout_ta(f)[0].squeeze(-2)  # [B,L,D]
            # enc_output = torch.cat([tf, af], dim=2)  # [B,L,2D]
            log_prob = F.log_softmax(self.smax_fc(enc_output), 2)
            # if return_attns:
            #     return log_prob, enc_output, enc_slf_attn_list
            return log_prob, enc_output, enc_output.reshape(enc_output.size(0) * enc_output.size(1), -1)
