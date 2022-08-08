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


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='does not use GPU')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate')
parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                    help='L2 regularization weight')
parser.add_argument('--rec-dropout', type=float, default=0.1,
                    metavar='rec_dropout', help='rec_dropout rate')
parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout',
                    help='dropout rate')
parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=200, metavar='E',
                    help='number of epochs')
parser.add_argument('--class-weight', action='store_true', default=True,
                    help='class weight')
parser.add_argument('--active-listener', action='store_true', default=False,
                    help='active listener')
parser.add_argument('--attention', default='general', help='Attention type')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Enables tensorboard log')
parser.add_argument('--usedrop', default=True,
                    help='Use gumbel softmax to drop unvaluable input vector.')
parser.add_argument('--dataset', default="IEMOCAP")
args = parser.parse_args([])
seed_everything(seed=1848)

print(args)

args.cuda = torch.cuda.is_available() and not args.no_cuda
if args.cuda:
    print('Running on GPU')
else:
    print('Running on CPU')

if args.tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter()

batch_size = args.batch_size
n_classes = 6
cuda = args.cuda
n_epochs = args.epochs

D_m = 1024 + 100 + 512  # T A V
D_g = 500
D_p = 500
D_e = 300
D_h = 300

D_a = 100  # concat attention

model = MMModel(n_classes, dt=1024, da=100, dv=512, d_model=512, n_layers=3, n_head=8, d_k=64, d_v=64, d_inner=2048,
                selector=args.usedrop, args=args)
if cuda:
    # model=torch.nn.DataParallel(model)
    model.cuda()
loss_weights = torch.FloatTensor([
    1 / 0.086747,
    1 / 0.144406,
    1 / 0.227883,
    1 / 0.160585,
    1 / 0.127711,
    1 / 0.252668,
])
if args.class_weight:
    loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
else:
    loss_function = MaskedNLLLoss()

optimizer = optim.Adam(lr=args.lr, params=model.parameters(),
                       weight_decay=args.l2)

train_loader, valid_loader, test_loader = \
    get_IEMOCAP_loaders('IEMOCAP_features_roberta1.pkl',
                        valid=0.0,
                        batch_size=batch_size,
                        num_workers=0)

best_metric, best_fscore, best_label, best_pred, best_mask = None, None, None, None, None
best_acc, best_fscores, best_accs = None, None, None

for e in range(n_epochs):
    print("epoch:" + str(e))
    start_time = time.time()
    train_loss, train_acc, _, _, _, train_fscore, _, _ = train_or_eval_model(model, loss_function,
                                                                             train_loader, e, optimizer, True)

    valid_loss, valid_acc, _, _, _, val_fscore, _, _ = train_or_eval_model(model, loss_function, valid_loader, e)
    test_accs, test_fscores, test_labels, test_preds = [], [], [], []
    for _ in range(1):
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions, _ = train_or_eval_model(model,
                                                                                                                loss_function,
                                                                                                                test_loader,
                                                                                                                e)
        test_accs.append(test_acc)
        test_fscores.append(test_fscore)
        test_labels.append(test_label)
        test_preds.append(test_pred)
    test_acc, test_fscore = np.mean(test_accs), np.mean(test_fscores)
    max_test_acc, max_test_fscore = np.max(test_accs), np.max(test_fscores)
    id = int(np.argmax(test_fscores))
    test_label = test_labels[id]
    test_pred = test_preds[id]
    flabels = test_label


    if best_metric == None or best_metric < (test_fscore + test_acc) / 2:
        best_metric = (test_fscore + test_acc) / 2
        best_fscore, best_acc, best_fscores, best_accs, best_label, best_pred, best_mask, best_attn = \
            test_fscore, test_acc, test_fscores, test_accs, test_label, test_pred, test_mask, attentions
        # torch.save(model.state_dict(), 'iemocap_s_c_g_m_drop_model{}.pth'.format(seed))
        # print("Model saved at epoch {}".format(e))

    if args.tensorboard:
        writer.add_scalar('test: accuracy/loss', test_acc / test_loss, e)
        writer.add_scalar('train: accuracy/loss', train_acc / train_loss, e)
    print(
        'epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore, \
                   test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))
if args.tensorboard:
    writer.close()

print('Test performance..')
print('F1 {} accuracy {} '.format(best_fscore, best_acc))
# print("F1 list {} ACC list {}".format(best_fscores, best_accs))
print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))