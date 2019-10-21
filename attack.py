import torch

from models import *
import argparse

import torchvision
import torchvision.transforms as transforms

from dataloader import SubsetOfList

# def get_centroids():

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sidx', default=0, type=int, help='starting id of each class')
    parser.add_argument('--eidx', default=2400, type=int, help='ending id of each class')
    parser.add_argument('--train-dp', type=float, default=0)
    parser.add_argument('--droplayer', type=float, default=0)
    args = parser.parse_args()

    print("Preparing data...")
    # may use data augmentation to boost the results later
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_list = torch.load('./datasets/CIFAR10_TRAIN_Split.pth')['clean_train']
    trainset = SubsetOfList(train_list, transform=transform_train, start_idx=args.sidx, end_idx=args.eidx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

    cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1000, shuffle=False, num_workers=2)

    # load the pre-trained models
    net = eval(args.net)(train_dp=args.train_dp, droplayer=args.droplayer)
