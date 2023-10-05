from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
from sklearn.linear_model import LogisticRegression
from utils import load_features
import pandas as pd

parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
parser.add_argument('--data-dir', type=str, required=True, help='data directory')
parser.add_argument('--result-dir', type=str, default='final_results', help='directory for saving results')
parser.add_argument('--extractor', type=str, default='resnet50', help='extractor type')
parser.add_argument('--dataset', type=str, default='SVHN', help='dataset')
parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
parser.add_argument('--std', type=float, default=10.0, help='standard deviation for objective perturbation')
parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
parser.add_argument('--train-splits', type=int, default=1, help='number of training data splits')
parser.add_argument('--subsample-ratio', type=float, default=1.0, help='negative example subsample ratio')
parser.add_argument('--num-steps', type=int, default=100, help='number of optimization steps')
parser.add_argument('--train-mode', type=str, default='ovr', help='train mode [ovr/binary]')
parser.add_argument('--train-sep', action='store_true', default=False, help='train binary classifiers separately')
parser.add_argument('--verbose', action='store_true', default=False, help='verbosity in optimizer')
parser.add_argument('--random_state', type=int, default=0, help='random seed')
parser.add_argument('--sampling', type=str, default='random', help='sampling')

args = parser.parse_args()

device = torch.device("cuda")
torch.manual_seed(args.random_state)

def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2

def lr_eval(w, X, y):
    return X.mv(w).sign().eq(y).float().mean()

def lr_grad(w, X, y, lam):
    grad = None
    for i in range(len(X)):
        z = torch.sigmoid(y * X.mv(w))
        if grad is None:
            grad = X.t().mv((z-1) * y) + lam * X.size(0) * w
        else:
            grad += X.t().mv((z-1) * y) + lam * X.size(0) * w
    return grad

def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()

def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose:
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data

# loads extracted features
X_train, X_test, y_train, y_train_onehot, y_test = load_features(args)
X_test = X_test.float().to(device)
y_test = y_test.to(device)

save_path = './%s/%s_%s_splits_%d_ratio_%.4f_std_%.1f_lam_%.0e.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)
if os.path.exists(save_path):
    # load trained models
    checkpoint = torch.load(save_path)
    w = checkpoint['w']
    weight = checkpoint['weight']
else:
    # train removal-enabled linear model
    weight = None
    start = time.time()
    # sample objective perturbation vector
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    y_train_onehot = y_train_onehot.float().to(device)
    w = lr_optimize(X_train, y_train, args.lam, b=None, num_steps=args.num_steps, verbose=args.verbose)
    print('Time elapsed: %.2fs' % (time.time() - start))
    torch.save({'w': w, 'weight': weight}, save_path)

pred = X_test.mv(w)
print('Test accuracy = %.4f' % pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean())

times = torch.zeros(args.num_removes)
accuracies = torch.zeros(args.num_removes)
losses = torch.zeros(args.num_removes)
w_approx = w.clone()
removals = torch.zeros(args.num_removes).type(torch.IntTensor)
n = len(X_train)
def return_loo_data(removal_indices, X_train, y_train):
    full_indices = set(range(n))
    removal_indices = set(removal_indices)
    keep_indices = torch.Tensor(list(full_indices.difference(removal_indices))).type(torch.int64)
    keep_indices = keep_indices.type(torch.int64)
    X_rem = torch.index_select(X_train, 0, keep_indices).float().to(device)
    y_rem = torch.index_select(y_train, 0, keep_indices).float().to(device)
    return X_rem, y_rem

print('Testing removal')

for i in range(args.num_removes):
        # removal from a single binary logistic regression model
    if args.sampling == 'adversarial':
        label_indices = np.where(y_train.cpu() == 1, 0.1, 0.9)
        label_indices = label_indices / np.sum(label_indices)
        removal_index = np.random.choice(list(range(len(X_train))), size=1, replace=False, p=label_indices)[0]
    else:
        removal_index = np.random.choice(list(range(len(X_train))), size=1, replace=False)[0]
    removals[i] = removal_index
    X_train = X_train.float().to('cpu')
    y_train = y_train.float().to('cpu')
    X_rem, y_rem = return_loo_data(removals, X_train, y_train)
    deletes_X = torch.index_select(X_train, 0, removals).float().to(device)
    deletes_y = torch.index_select(y_train, 0, removals).float().to(device)
    print(len(deletes_X))
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    rem_len = len(removals)
    start = time.time()
    H_inv = (1 / (n-rem_len)) * lr_hessian_inv(w[:], X_rem.float().to(device), y_rem.float().to(device), args.lam)
    grad_deletes = lr_grad(w, deletes_X, deletes_y, args.lam)
    Delta = H_inv.mv(grad_deletes)
    b = args.std * torch.randn(X_train.size(1)).float().to(device)
    w_approx = w_approx + (1 / (n-rem_len)) * Delta
    w_approx += b
    times[i] = time.time() - start
    print('Iteration %d: time = %.2fs' % (i+1, times[i]))
            
    pred = X_test.mv(w_approx)
    accuracies[i] = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean()
    print('Test accuracy = %.4f' % accuracies[i])

    losses[i] = lr_loss(w_approx, X_test, y_test, args.lam)
    print('Test Loss = %.4f' % losses[i])

save_path = './%s/%s_%s_splits_%d_ratio_%.4f_std_%.1f_lam_%.0e_removal.pth' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam)
torch.save({'times': times, 'accuracies': accuracies, 'losses': losses}, save_path)
results_df = pd.DataFrame(columns=['Delete Request', 'Random State', 'Time', 'Test Acc', 'Test Loss'])
results_df['Delete Request'] = np.array(list(range(args.num_removes)))
results_df['Time'] = times.numpy()
results_df['Test Acc'] = accuracies.numpy()
results_df['Test Loss'] = losses.numpy()
results_df['Random State'] = args.random_state
results_path = './%s/%s_%s_splits_%d_ratio_%.4f_std_%.1f_lam_%.0e_num_delete_%d_removal_sekhari_sampling_%s_%d.csv' % (
    args.result_dir, args.extractor, args.dataset, args.train_splits, args.subsample_ratio, args.std, args.lam, args.num_removes, args.sampling, args.random_state)
results_df.to_csv(results_path)