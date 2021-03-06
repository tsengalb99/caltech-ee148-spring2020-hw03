from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import random
import os
import sys
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 48, 2, stride=1, padding=2)

        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)

        self.dropout_fc = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(3072, 400)
        self.fc3 = nn.Linear(400, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()   # Set the model to training mode
    train_loss = 0
    train_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        
        train_loss += F.nll_loss(output, target, reduction='sum').item()
        train_num += len(data)
        
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.sampler),
        #        100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/train_num

def test(model, device, test_loader, valid=True):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.02f}%)'.format(
        "Test" if not valid else "Valid", 
        test_loss, correct, test_num,
        100. * correct / test_num))
    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--frac', type=float, default=1.0)
    
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--save-path', type=str, default='../models/mnist')
    parser.add_argument('--save-name', type=str,
                        help='model save path')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 12, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ColorJitter(brightness=0.25, contrast=0.25),
                    transforms.RandomAffine(degrees=4, shear=6),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    valid_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    
    test_dataset = datasets.MNIST('../data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2048, shuffle=False, **kwargs)

    inds = [[] for _ in range(len(set(train_dataset.targets)))]
    for i in range(len(train_dataset.targets)):
        inds[train_dataset.targets[i]].append(i)

    for i in range(len(inds)):
        random.shuffle(inds[i])

    valid = []
    for i in inds:
        valid += i[0:int(0.15*len(i))]
    assert(len(valid) == len(set(valid)))
    valid = set(valid)
    
    total = set(range(len(train_dataset)))
    subset_indices_train = list(total.difference(valid))
    random.shuffle(subset_indices_train)
    subset_indices_train = subset_indices_train[0:int(args.frac*len(subset_indices_train))]
    subset_indices_valid = list(valid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train),
        **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=4096,
        sampler=SubsetRandomSampler(subset_indices_valid),
        **kwargs
    )

    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    loss_train = []
    loss_valid = []
    
    for epoch in range(1, args.epochs + 1):
        loss_train.append(train(args, model, device, train_loader, optimizer, epoch))
        loss_valid.append(test(model, device, val_loader, valid=True))
        scheduler.step()    # learning rate scheduler

    test(model, device, test_loader, valid=False)
        
    path = os.path.join(args.save_path, args.save_name)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    np.array(loss_train).dump(os.path.join(path, "train_acc.npy"))
    np.array(loss_valid).dump(os.path.join(path, "valid_acc.npy"))
    os.system('cp {} {}'.format(sys.argv[0], os.path.join(path, sys.argv[0])))

if __name__ == '__main__':
    main()
