import argparse
import shutil
import time
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import math
import sys
import random
from models import *


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--test', default='', type=str, metavar='PATH',
                    help='path to pre-trained model (default: none)')
parser.add_argument('--type', default='', type=str, help='choose dataset (cifar10, cifar100)')
parser.add_argument('--model', default='', type=str, help='choose model type (resnet)')
# for resnet, wideresnet
parser.add_argument('--depth', type=int, default=0, help='model depth for resnet')
# index of each training runs
parser.add_argument('--tn', type=str, default='', help='n-th training')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
best_prec1 = 0


def main():
    global args, best_prec1
    model_name = ''
    class_num = 0
    args = parser.parse_args()
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # trained model test code
    if args.test != '':
        print("=> Testing trained weights ")
        checkpoint = torch.load(args.test)
        print("=> loaded test checkpoint: {} epochs, Top1 Accuracy: {}, Top5 Accuracy: {}".format(checkpoint['epoch'],
                                                                                                  checkpoint[
                                                                                                      'test_acc1'],
                                                                                                  checkpoint[
                                                                                                      'test_acc5']))
        return
    else:
        print("=> No Test ")

    # data loader setting
    if args.type == 'cifar10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(), transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR10(root='./dataset/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./dataset/', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        class_num = 10
    elif args.type == 'cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR100(root='./dataset/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./dataset/', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
        class_num = 100
    else:
        print("No dataset")

    # create model
    if args.model == 'resnet':
        cifar_list = [20, 32, 44, 56, 110]
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)')
        if args.depth in cifar_list:
            assert (args.depth - 2) % 6 == 0
            n = int((args.depth - 2) / 6)
            model = ResNet_Cifar(BasicBlock, [n, n, n], num_classes=class_num)
        else:
            print("Inappropriate ResNet model")
            return
        model_name = args.model+str(args.depth)
    else:
        print("No model")
        return

    num_parameters = sum(l.nelement() for l in model.parameters())
    num_parameters = round((num_parameters / 1e+6), 3)
    print("model name : ", model_name)
    print("model parameters : ", num_parameters, "M")
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # make progress save directory
    save_progress = './checkpoints/' + args.type + '/' + model_name + '/Baseline/' + str(args.tn)
    if not os.path.isdir(save_progress):
        os.makedirs(save_progress)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # adjust_learning_rate(auto_optimizer, epoch)

        tr_acc, tr_acc5, tr_loss = train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1, prec5, te_loss = test(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'train_fc_loss': tr_loss, 'test_fc_loss': te_loss,
                         'train_acc1': tr_acc, 'train_acc5': tr_acc5, 'test_acc1': prec1, 'test_acc5': prec5}, is_best, save_progress)
        torch.save(model.state_dict(), save_progress + '/weight.pth')
        if is_best:
            torch.save(model.state_dict(), save_progress + '/best_weight.pth')

    print('Best accuracy (top-1):', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # total loss
    losses = AverageMeter()
    # performance
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # baseline training
    return top1.avg, top5.avg, losses


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # total loss
    losses = AverageMeter()
    # performance
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # baseline training
    return top1.avg, top5.avg, losses


def save_checkpoint(state, is_best, save_path):
    save_dir = save_path
    torch.save(state, save_path + '/' + str(state['epoch']) + 'epoch_result.pth')
    if is_best:
        torch.save(state, save_dir + '/best_result.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
