import torch
import time
import math
import os
import sys
from torchvision import datasets, transforms
from warfarin_preprocess import preprocess_data
from sklearn.model_selection import train_test_split
import numpy as np



# extracts features into a tensor
def extract_features(extr, device, data_loader):
    extr.eval()
    features = None
    labels = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = extr(data).data.cpu()
            if features is None:
                features = output.squeeze()
                labels = target
            else:
                features = torch.cat([features, output.squeeze()], dim=0)
                labels = torch.cat([labels, target], dim=0)
    return features, labels

# constructs one-hot representations of labels
def onehot(y):
    y_onehot = -torch.ones(y.size(0), y.max() + 1).float()
    y_onehot.scatter_(1, y.long().unsqueeze(1), 1)
    return y_onehot

# loads features from a saved checkpoint or directly as raw features
def load_features(args):

    if args.dataset == 'SVHN':  
        ckpt_file = './data/dp_delta_1.00e-05_std_6.00_SVHN_extracted.pth'
        if os.path.exists(ckpt_file):
            checkpoint = torch.load(ckpt_file)
            X_train = checkpoint['X_train'].cpu()
            y_train = checkpoint['y_train'].cpu()
            X_test = checkpoint['X_test'].cpu()
            y_test = checkpoint['y_test'].cpu()
            train_indices = (y_train.eq(3) + y_train.eq(8)).gt(0)
            test_indices = (y_test.eq(3) + y_test.eq(8)).gt(0)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices].eq(3).float()
            X_test = X_test[test_indices]
            y_test = y_test[test_indices].eq(3).float()
    elif args.dataset == 'MNIST':
            trainset = datasets.MNIST(args.data_dir, download=True, train=True, transform=transforms.ToTensor())
            testset = datasets.MNIST(args.data_dir, download=True, train=False, transform=transforms.ToTensor())
            X_train = torch.zeros(len(trainset), 784)
            y_train = torch.zeros(len(trainset))
            X_test = torch.zeros(len(testset), 784)
            y_test = torch.zeros(len(testset))
            for i in range(len(trainset)):
                x, y = trainset[i]
                X_train[i] = x.view(784) - 0.5
                y_train[i] = y
            for i in range(len(testset)):
                x, y = testset[i]
                X_test[i] = x.view(784) - 0.5
                y_test[i] = y
            # load classes 3 and 8
            train_indices = (y_train.eq(3) + y_train.eq(8)).gt(0)
            test_indices = (y_test.eq(3) + y_test.eq(8)).gt(0)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices].eq(3).float()
            X_test = X_test[test_indices]
            y_test = y_test[test_indices].eq(3).float()
            # L2 normalize features
            X_train /= X_train.norm(2, 1).unsqueeze(1)
            X_test /= X_test.norm(2, 1).unsqueeze(1)
    elif args.dataset == 'Warfarin':
        X, y = preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(torch.Tensor(X), torch.Tensor(y), test_size=0.33, random_state=42)
                    # L2 normalize features
        X_train /= X_train.norm(2, 1).unsqueeze(1)
        X_test /= X_test.norm(2, 1).unsqueeze(1)
        y_train_onehot = None

    else:
        print("Error: Unknown dataset %s. Aborting." % args.dataset) 
        sys.exit(1)
        
    
    # convert labels to +/-1 or one-hot vectors
    if args.train_mode == 'binary':
        y_train_onehot = y_train
        y_train = (2 * y_train - 1)
    else:
        y_train_onehot = onehot(y_train)
    if y_train_onehot is not None:
        if len(y_train_onehot.size()) == 1:
            y_train_onehot = y_train_onehot.unsqueeze(1)
        
    return X_train, X_test, y_train, y_train_onehot, y_test

def soft_thresh(x, l):
  return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def fista(H, g, lam, maxit, x0 = None):
    # Returns argmin_b <g, b> + (1/2) b H b + lam||b||_1
    # 
    # Args:
    #   x0 - optional initial value of optimization vector
    H = H.cpu().numpy()
    g = g.cpu().numpy()
    x0 = x0.cpu().numpy()
    start = time.time()
    if x0 is None:
      print("hello")
      x = np.zeros(H.shape[1])
    else:
      x = x0.copy()
    pobj = []
    t = 1
    z = x.copy()
    L = np.linalg.norm(H)
    for iter in range(int(maxit)):
      xold = x.copy()
      z = z - (g + H.dot(z))/L
      x = soft_thresh(z, lam / L)
      t0 = t
      t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
      z = x + ((t0 - 1.) / t) * (x - xold)
      this_pobj = 0.5 * (H.dot(x).dot(x)) + g.dot(x) + lam * np.linalg.norm(x, 1)
      if iter % 100 == 0:
        print(this_pobj)
      #pobj.append(this_pobj))
      #pobj.append((time.time() - time0, this_pobj))
    print("Fit fista in {}s".format(time.time() - start))
    #times, pobj = map(np.array, zip(*pobj))
    return torch.Tensor(x)