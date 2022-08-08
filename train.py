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
from model import *


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


# def get_train_valid_sampler(trainset, valid=0.1):
#     size = len(trainset)
#     idx = list(range(size))
#     split = int(valid*size)
#     return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_train_valid_sampler(trainset, start=1039, end=1153):
    size = len(trainset)
    idx = list(range(size))
    # split = int(valid*size)
    return SubsetRandomSampler(idx[:start]), SubsetRandomSampler(idx[start:end])


def get_MELD_loaders(path, n_classes, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset(path=path, n_classes=n_classes)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset)
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

    testset = MELDDataset(path=path, n_classes=n_classes, train=False)
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
    fs = []
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
        textf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        # Subsequence mask.
        submask = (1 - torch.triu(torch.ones((1, umask.size(1), umask.size(1)), device=umask.device),
                                  diagonal=1)).bool()
        # Padding mask.
        padmask = umask.unsqueeze(-2).bool()
        # Self(Speaker2Speaker) mask.
        smask = (qmask[:, :, 0] * 2 + qmask[:, :, 1]).transpose(0, 1)  # [B,L]
        smask = smask.unsqueeze(1).expand(-1, smask.size(1), -1)  # [B,L,L]
        smask = (smask.transpose(1, 2) == smask).bool()  # [B,L,L]
        # Cross(Speaker2Listener) mask.
        cmask = (qmask[:, :, 0] * 2 + qmask[:, :, 1]).transpose(0, 1)  # [B,L]
        cmask = cmask.unsqueeze(1).expand(-1, cmask.size(1), -1)  # [B,L,L]
        cmask = (cmask.transpose(1, 2) == cmask).bool()  # [B,L,L]
        # Final mask.
        smask = submask & padmask & smask
        cmask = submask & padmask & cmask
        smask = smask.cuda() if cuda else smask
        cmask = cmask.cuda() if cuda else cmask

        log_prob, alpha, f = model(textf.transpose(0, 1), acouf.transpose(0, 1), None, smask,
                                   cmask)  # batch, seq_len, n_classes
        lp_ = log_prob.view(-1, log_prob.size()[2])  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        fs.append(f.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            #             if args.tensorboard:
            #                 for param in model.named_parameters():
            #                     writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            pass

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
        # fs = np.concatenate(fs,axis=0)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    class_report = classification_report(labels, preds, sample_weight=masks, digits=4)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b,
                                                                      vids], class_report, fs


    seed_everything(seed = 72)
    cuda = torch.cuda.is_available()
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # choose between 'sentiment' or 'emotion'
    classification_type = 'emotion'
    feature_type = 'all'

    data_path = ''
    batch_size = 30
    n_classes = 3
    n_epochs = 50
    active_listener = False
    attention = 'general'
    class_weight = False
    dropout = 0.0
    rec_dropout = 0.0
    l2 = 0.00001
    lr = 0.000005

    if feature_type == 'text':
        print("Running on the text features........")
        D_m = 600 + 600
    elif feature_type == 'audio':
        print("Running on the audio features........")
        D_m = 300 + 300
    else:
        print("Running on the multimodal features........")
        D_m = 1024 + 300
    D_g = 150
    D_p = 150
    D_e = 100
    D_h = 100

    D_a = 100  # concat attention

    loss_weights = torch.FloatTensor([1.0, 1.0, 1.0])

    if classification_type.strip().lower() == 'emotion':
        n_classes = 7
        loss_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="MELD")
    args = parser.parse_args([])

    model = SMModel(n_classes, dt=1024, da=300, dv=1, d_model=512, n_layers=3, n_head=8, d_k=64, d_v=64, d_inner=2048,
                    selector=True, args=args)
    # model.load_state_dict(torch.load("meld_s_c_g_m_drop_model101.pth"))

    if cuda:
        model.cuda()
    if class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    # strategies = []
    # for name, param in  model.named_parameters():
    #     if ("w_ds0" in name or "w_ds1" in name):
    #         strategies.append({"params":param, "lr":lr * 1e3})
    #     else:
    #         strategies.append({"params": param, "lr": lr})
    # optimizer = optim.AdamW(strategies,
    #                        weight_decay=l2)

    train_loader, valid_loader, test_loader = \
        get_MELD_loaders(data_path + 'MELD_features_roberta1.pkl', n_classes,
                         valid=0.0,
                         batch_size=batch_size,
                         num_workers=0)

    best_metric, best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None, None

    for e in range(n_epochs):
        print(e)
        strategies = []
        for name, param in model.named_parameters():
            if ("w_ds0" in name or "w_ds1" in name):
                strategies.append({"params": param, "lr": lr * min(1e1 * e, 1e3)})
            else:
                strategies.append({"params": param, "lr": lr})
        optimizer = optim.AdamW(strategies,
                                weight_decay=l2)

        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _, _ = train_or_eval_model(model, loss_function,
                                                                                    train_loader, e, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_fscore, _, _, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, test_class_report, fs = train_or_eval_model(
            model, loss_function, test_loader, e)
        # flabels=test_label

        if best_metric == None or best_metric > (valid_loss) / 1:
            best_metric = (valid_loss) / 1
            best_fscore, best_loss, best_label, best_pred, best_mask, best_attn = \
                test_fscore, test_loss, test_label, test_pred, test_mask, attentions
            # torch.save(model.state_dict(), 'meld_s_c_g_m_drop_model{}_3.pth'.format(iiii))

        # if args.tensorboard:
        #     writer.add_scalar('test: accuracy/loss',test_acc/test_loss,e)
        #     writer.add_scalar('train: accuracy/loss',train_acc/train_loss,e)

        print(
            'epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore, \
                   test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))

    print('Test performance..')
    print('Fscore {} accuracy {}'.format(best_fscore,
                                         round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100,
                                               2)))
    print('{} {} {}'.format(seed, best_fscore,
                            round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))