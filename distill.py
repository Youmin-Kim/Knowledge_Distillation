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
from scipy.stats import norm


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
                    help='path to trained model (default: none)')
parser.add_argument('--type', default='', type=str, help='choose dataset cifar10, cifar100, imagenet')
parser.add_argument('--teacher', default='', type=str, help='pre-trained teacher network type (resnet, wideresnet)')
parser.add_argument('--student', default='', type=str, help='to be trained student network type (resnet, wideresnet)')
# for teacher
# for all resnet
parser.add_argument('--depth', type=int, default=0, help='depth for resnet, wideresnet')
# for wideresnet
parser.add_argument('--wfactor', type=int, default=1, help='wide factor for wideresnet')
# index of each training runs
parser.add_argument('--tn', type=str, default='', help='n-th training')
# for student
# for all resnet
parser.add_argument('--sdepth', type=int, default=0, help='depth for resnet, wideresnet')
# for wideresnet
parser.add_argument('--swfactor', type=int, default=1, help='wide factor for wideresnet')
# index of each training runs
parser.add_argument('--stn', type=str, default='', help='n-th training')
# distillation method for training student
parser.add_argument('--distype', type=str, default='', help='distillation type, empty means exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
best_prec1 = 0


def main():
    global args, best_prec1
    teacher_name = ''
    student_name = '_distilled_by_'
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
        trainset = datasets.CIFAR10(root='/dataset/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='/dataset/', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        class_num = 10
    elif args.type == 'cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4814, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        trainset = datasets.CIFAR100(root='/dataset/CIFAR', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='/dataset/CIFAR', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
        class_num = 100
    else:
        print("No dataset")

    BB = BasicBlock
    if args.distype == 'OD':
        BB = BasicBlock_BN

    # load the pre-trained teacher
    if args.teacher == 'resnet':
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)\n'
              'ImageNet 18(11.68M), 34(21.79M), 50(25.5M)')
        cifar_list = [20, 32, 44, 56, 110]
        # CIFAR10, CIFAR100
        if args.depth in cifar_list:
            assert (args.depth - 2) % 6 == 0
            n = int((args.depth - 2) / 6)
            teacher = ResNet_Cifar(BB, [n, n, n], num_classes=class_num)
        else:
            print("Inappropriate ResNet Teacher model")
            return
        teacher_name = args.teacher+str(args.depth)
    elif args.teacher == 'wideresnet':
        print('WideResNet CIFAR10, CIFAR100 : 40_1(0.6M), 40_2(2.2M), 40_4(8.9M), 40_8(35.7M), 28_10(36.5M), 28_12(52.5M),'
            ' 22_8(17.2M), 22_10(26.8M), 16_8(11.0M), 16_10(17.1M)')
        assert (args.depth - 4) % 6 == 0
        n = int((args.depth - 4) / 6)
        teacher = Wide_ResNet_Cifar(BB, [n, n, n], wfactor=args.wfactor, num_classes=class_num)
        teacher_name = args.teacher+str(args.depth)+'_'+str(args.wfactor)
    else:
        print("No Teacher model")
        return

    # create student
    if args.student == 'resnet':
        print('ResNet CIFAR10, CIFAR100 : 20(0.27M) 32(0.46M), 44(0.66M), 56(0.85M), 110(1.7M)\n'
              'ImageNet 18(11.68M), 34(21.79M), 50(25.5M)')
        cifar_list = [20, 32, 44, 56, 110]
        # CIFAR10, CIFAR100
        if args.sdepth in cifar_list:
            assert (args.sdepth - 2) % 6 == 0
            n = int((args.sdepth - 2) / 6)
            student = ResNet_Cifar(BB, [n, n, n], num_classes=class_num)
        else:
            print("Inappropriate ResNet Student model")
            return
        student_name = args.student + str(args.sdepth) + student_name + teacher_name + '_' + str(args.tn) + 'th'
    elif args.student == 'wideresnet':
        print('WideResNet CIFAR10, CIFAR100 : 40_1(0.6M), 40_2(2.2M), 40_4(8.9M), 40_8(35.7M), 28_10(36.5M), 28_12(52.5M),'
            ' 22_8(17.2M), 22_10(26.8M), 16_8(11.0M), 16_10(17.1M)')
        assert (args.sdepth - 4) % 6 == 0
        n = int((args.sdepth - 4) / 6)
        student = Wide_ResNet_Cifar(BB, [n, n, n], wfactor=args.swfactor, num_classes=class_num)
        student_name = args.student + str(args.sdepth) + '_' + str(args.swfactor) + student_name + teacher_name + '_' + str(args.tn) + 'th'
    else:
        print("No Student model")
        return

    # print pre-trained teacher and to-be-trained student information
    t_num_parameters = round((sum(l.nelement() for l in teacher.parameters()) / 1e+6), 3)
    s_num_parameters = round((sum(l.nelement() for l in student.parameters()) / 1e+6), 3)
    print("teacher name : ", teacher_name)
    print("teacher parameters : ", t_num_parameters, "M")
    print("student name : ", student_name)
    print("student parameters : ", s_num_parameters, "M")
    teacher = torch.nn.DataParallel(teacher).cuda()
    load_teacher_progress = './checkpoint/' + args.type + '/' + teacher_name + '/Baseline/' + str(args.tn)
    teacher.load_state_dict(torch.load(load_teacher_progress + '/best_weight.pth'))
    student = torch.nn.DataParallel(student).cuda()

    # define optimizer or loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # check the performance of the pre-trained teacher
    print("check the performance of the pre-trained teacher")
    t1, t5, _ = test(val_loader, teacher, criterion)
    print("pre-trained teacher (top1, top5) : ", t1, t5)

    # add the additional weights for some distillation methods
    if args.distype == 'FN':
        rand_data = torch.randn(args.batch_size, 3, 32, 32)
        teacher.eval()
        student.eval()
        _, t_in_feat = teacher(rand_data, type=args.distype)
        _, s_in_feat = student(rand_data, type=args.distype)
        trainable_list = nn.ModuleList([])
        trainable_list.append(student)
        regressor = ConvReg(s_shape=s_in_feat[1].shape, t_shape=t_in_feat[1].shape)
        regressor = torch.nn.DataParallel(regressor).cuda()
        trainable_list.append(regressor)
        optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.distype == 'OD':
        trainable_list = nn.ModuleList([])
        trainable_list.append(student)
        t_ch_l = [16 * args.wfactor, 32 * args.wfactor, 64 * args.wfactor]
        s_ch_l = [16 * args.swfactor, 32 * args.swfactor, 64 * args.swfactor]
        regressor = []
        for t, s in zip(t_ch_l, s_ch_l):
            temp_regressor = build_feature_connector(t, s)
            temp_regressor = torch.nn.DataParallel(temp_regressor).cuda()
            regressor.append(temp_regressor)
            trainable_list.append(temp_regressor)
        optimizer = torch.optim.SGD(trainable_list.parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(student.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        regressor = []

    # make progress save directory
    save_progress = './checkpoints/' + args.type + '/' + student_name + '/' + args.distype + '/' + str(args.stn)
    if not os.path.isdir(save_progress):
        os.makedirs(save_progress)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        tr_acc, tr_acc5, tr_fc_loss, tr_d_loss = distillation(train_loader, teacher, student, criterion, optimizer, epoch, regressor)
        # evaluate on validation set
        prec1, prec5, te_fc_loss = test(val_loader, student, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({'epoch': epoch + 1, 'train_fc_loss': tr_fc_loss, 'train_d_loss': tr_d_loss, 'test_fc_loss': te_fc_loss,
                         'train_acc1': tr_acc, 'train_acc5': tr_acc5, 'test_acc1': prec1, 'test_acc5': prec5}, is_best, save_progress)
        torch.save(student.state_dict(), save_progress + '/weight.pth')
        if is_best:
            torch.save(student.state_dict(), save_progress + '/best_weight.pth')

    print('Best accuracy (top-1):', best_prec1)


def distillation(train_loader, teacher, student, criterion, optimizer, epoch, reg=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    dis_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    student.train()
    if args.distype != 'OD':
        teacher.eval()
    else:
        teacher.train()
    end = time.time()
    loss = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()

        # distilling
        if args.distype == 'KD':
            alpha = 0.9
            T = 4
            t_output = teacher(input, type=args.distype)
            s_output = student(input, type=args.distype)
            kd_loss = F.kl_div(F.log_softmax(s_output / T, dim=1), F.softmax(t_output / T, dim=1), reduction='batchmean') * (T ** 2)
            ce_loss = criterion(s_output, target)
            loss = alpha * kd_loss + (1. - alpha) * ce_loss
            dis_loss = kd_loss
        elif args.distype == 'FN':
            beta = 100
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            fitnet_loss = nn.MSELoss()
            loss = criterion(s_output, target) + beta * fitnet_loss(reg(s_middle_output[1]), t_middle_output[1].detach())
            dis_loss = fitnet_loss(reg(s_middle_output[1]), t_middle_output[1].detach())
        elif args.distype == 'AT':
            beta = 1e+3
            att_loss = 0
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            for k in range(len(t_middle_output)):
                att_loss += attention_loss(t_middle_output[k].detach(), s_middle_output[k])
            ce_loss = criterion(s_output, target)
            loss = ce_loss + (beta / 2) * att_loss
            dis_loss = att_loss
        elif args.distype == 'NST':
            beta = 50
            nst_loss = 0
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            for k in range(len(t_middle_output)):
                nst_loss += poly_kernel_loss(s_middle_output[k], t_middle_output[k].detach())
            ce_loss = criterion(s_output, target)
            loss = ce_loss + (beta / 2) * nst_loss
            dis_loss = nst_loss
        elif args.distype == 'OD':
            beta = 1e+3
            od_loss = 0
            teacher_bns = teacher.module.get_bn_before_relu()
            margins = [get_margin_from_BN(bn) for bn in teacher_bns]
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            num_reg = len(reg)
            for k in range(num_reg):
                od_loss += (mReLU_loss(reg[k](s_middle_output[k]), t_middle_output[k].detach(),
                                      margins[k].unsqueeze(1).unsqueeze(2).unsqueeze(0).detach()) / (2 ** (num_reg - k - 1)))
            od_loss = od_loss.sum() / input.size()[0]
            ce_loss = criterion(s_output, target)
            loss = ce_loss + (1 / beta) * od_loss
            dis_loss = od_loss
        elif args.distype == 'RKD':
            h_d = 25.
            h_a = 50.
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            t_middle_output = t_middle_output.view(input.size(0), -1)
            s_middle_output = s_middle_output.view(input.size(0), -1)
            # RKD distance loss
            with torch.no_grad():
                t_d = pdist(t_middle_output, squared=False)
                mean_td = t_d[t_d > 0].mean()
                t_d = t_d / mean_td
            d = pdist(s_middle_output, squared=False)
            mean_d = d[d > 0].mean()
            d = d / mean_d
            loss_d = F.smooth_l1_loss(d, t_d)

            # RKD Angle loss
            with torch.no_grad():
                td = (t_middle_output.unsqueeze(0) - t_middle_output.unsqueeze(1))
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

            sd = (s_middle_output.unsqueeze(0) - s_middle_output.unsqueeze(1))
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
            loss_a = F.smooth_l1_loss(s_angle, t_angle)
            ce_loss = criterion(s_output, target)
            loss = ce_loss + h_d * loss_d + h_a * loss_a
            dis_loss = loss_d + loss_a
        elif args.distype == 'SP':
            gamma = 3e+3
            t_output, t_middle_output = teacher(input, type=args.distype)
            s_output, s_middle_output = student(input, type=args.distype)
            sp_loss = similarity_preserve_loss(t_middle_output[2].detach(), s_middle_output[2])
            ce_loss = criterion(s_output, target)
            loss = ce_loss + gamma * sp_loss
            dis_loss = sp_loss
        else:
            print("No Distillation")
            return

        # measure accuracy and record loss
        prec1, prec5 = accuracy(s_output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        ce_losses.update(ce_loss.item(), input.size(0))
        dis_losses.update(dis_loss.item(), input.size(0))

        # compute gradient and do SGD step
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

    return top1.avg, top5.avg, ce_losses, dis_losses


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()

        # compute output
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    return top1.avg, top5.avg, losses


def attention(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_loss(t, s):
    return (attention(t) - attention(s)).pow(2).mean()


def poly_kernel(a, b):
    a = a.unsqueeze(1)
    b = b.unsqueeze(2)
    res = (a * b).sum(-1).pow(2)
    return res


def poly_kernel_loss(f_s, f_t):
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass

    f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
    f_s = F.normalize(f_s, dim=2)
    f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
    f_t = F.normalize(f_t, dim=2)

    # set full_loss as False to avoid unnecessary computation
    full_loss = True
    if full_loss:
        return (poly_kernel(f_t, f_t).mean().detach() + poly_kernel(f_s, f_s).mean()
                - 2 * poly_kernel(f_s, f_t).mean())
    else:
        return poly_kernel(f_s, f_s).mean() - 2 * poly_kernel(f_s, f_t).mean()


def mReLU_loss(source, target, margin):
    loss = ((source - margin)**2 * ((source > margin) & (target <= margin)).float() +
            (source - target)**2 * ((source > target) & (target > margin) & (target <= 0)).float() +
            (source - target)**2 * (target > 0).float())
    return torch.abs(loss).sum()


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def similarity_preserve_loss(t, s):
    bsz = s.size()[0]
    f_s = s.view(bsz, -1)
    f_t = t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
