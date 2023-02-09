#!/usr/bin/env python3
from __future__ import division
import time
import argparse
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.BowelDatasetFineSeg import BowelFineSeg
from tools.loss import *
import os, math
import shutil
import seg_model

import wandb


os.environ['CUDA_VISIBLE_DEVICES'] = '1'   # "0, 1"
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        # nn.init.kaiming_normal_(m.weight)
        nn.init.xavier_normal_(m.weight, gain=0.02)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)    # 4 (single GPU)
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--nEpochs_base', type=int, default=501, help='total epoch number for base segmentor')
    parser.add_argument('--nEpochs_meta', type=int, default=301, help='total epoch number for meta segmentor')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--eval_interval', default=5, type=int, help='evaluate interval on validation set')
    parser.add_argument('--temp', default=0.7, type=float, help='temperature for meta segmentor')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--deterministic', type=bool, default=True)
    args = parser.parse_args()


    args.save_dir = 'exp/BowelNet.{}'.format(datestr())
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'infer.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'seg_model.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'tools/loss.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'tools/utils.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'datasets/BowelDatasetFineSeg.py'), dst=args.save_dir)


    # if args.seed is not None:
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = args.deterministic


    print("build BowelNet")
    model = seg_model.BowelNet(elu=False)
    model.apply(weights_init)



    # model.load_state_dict(torch.load("./BowelNet.20230207_1744/rectum_meta_195.pth"))




    model = model.cuda()
    model = nn.parallel.DataParallel(model)


    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))






    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_small/'
    # bowel_name = 'small'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_colon/'
    # bowel_name = 'colon'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_sigmoid/'
    # bowel_name = 'sigmoid'

    # data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_duodenum/'
    # bowel_name = 'duodenum'

    data_dir = '/mnt/c/chong/data/Bowel/crop_stage1_ROI_rectum/'
    bowel_name = 'rectum'



    # WandB – Initialize a new run
    wandb.init(project='BowelNet', mode='disabled')     # mode='disabled'
    wandb.run.name = bowel_name + '_' + wandb.run.id



    print("loading training set")
    trainSet = BowelFineSeg(root=data_dir, mode="train", transform=True, bowel_name=bowel_name, save_dir=args.save_dir)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)  # 8

    print("loading test set")
    testSet = BowelFineSeg(root=data_dir, mode="test", transform=False, bowel_name=bowel_name, save_dir=args.save_dir)
    testLoader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)  # 8



    # base segmentor training
    ##########################################################################
    params_base = [param for name, param in model.named_parameters() if 'meta' not in name]
    optimizer_base = optim.Adam(params_base, lr=args.lr, weight_decay=args.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_base, step_size=50, gamma=0.85)
    best_dice = 0.0
    trainF_base = open(os.path.join(args.save_dir, 'train_base.csv'), 'w')
    testF_base = open(os.path.join(args.save_dir, 'test_base.csv'), 'w')
    for epoch in range(1, args.nEpochs_base + 1):
        train_base(epoch, model, trainLoader, optimizer_base, trainF_base, train_segmentor='base')
        # base_scheduler.step()
        wandb.log({
            "Base LR": optimizer_base.param_groups[0]['lr'],
        })
        if epoch % args.eval_interval == 0:  # 5
            dice = test_base(epoch, model, testLoader, testF_base, train_segmentor='base')
            is_best = False
            if dice > best_dice:
                is_best = True
                best_dice = dice
            save_checkpoint({'epoch': epoch, 'state_dict': model.module.state_dict(), 'best_acc': best_dice},
                            is_best, args.save_dir, bowel_name + '_base')
            torch.save(model.module.state_dict(), args.save_dir + '/' + bowel_name + '_base_' + str(epoch) + '.pth')
    trainF_base.close()
    testF_base.close()

    # best_base_model_path = os.path.join(args.save_dir, bowel_name + '_base' + '_model_best.pth.tar')
    # model.load_state_dict(torch.load(best_base_model_path)['state_dict'])

    # meta segmentor training
    ##########################################################################
    params_meta = [param for name, param in model.named_parameters() if 'meta' in name]
    optimizer_meta = optim.Adam(params_meta, lr=args.lr, weight_decay=args.weight_decay)
    meta_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=30, gamma=0.85)
    best_dice = 0.0
    trainF_meta = open(os.path.join(args.save_dir, 'train_meta.csv'), 'w')
    testF_meta = open(os.path.join(args.save_dir, 'test_meta.csv'), 'w')
    for epoch in range(1, args.nEpochs_meta + 1):
        train_meta(epoch, model, trainLoader, optimizer_meta, trainF_meta, train_segmentor='meta', temperature=args.temp)
        # meta_scheduler.step()
        wandb.log({
            "Meta LR": optimizer_meta.param_groups[0]['lr'],
        })
        if epoch % args.eval_interval == 0:  # 5
            dice = test_meta(epoch, model, testLoader, testF_meta, train_segmentor='meta')
            is_best = False
            if dice > best_dice:
                is_best = True
                best_dice = dice
            save_checkpoint({'epoch': epoch, 'state_dict': model.module.state_dict(), 'best_acc': best_dice},
                            is_best, args.save_dir, bowel_name + '_meta')
            torch.save(model.module.state_dict(), args.save_dir + '/' + bowel_name + '_meta_' + str(epoch) + '.pth')
    trainF_meta.close()
    testF_meta.close()



def train_base(epoch, model, trainLoader, optimizer, trainF, train_segmentor='base'):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target, edge, skele_x, skele_y, skele_z) in enumerate(trainLoader):
        data, target, edge, skele_x, skele_y, skele_z = Variable(data.cuda()), Variable(target.cuda()), \
                                                        Variable(edge.cuda()), Variable(skele_x.cuda()), \
                                                        Variable(skele_y.cuda()), Variable(skele_z.cuda())
        optimizer.zero_grad()
        out_flux_reg, out_skele_mask_seg, out_edge_reg, out_edge_mask_seg = model(data, train_segmentor)

        #####  boundary segmentor loss
        loss_ce_edge_mask = F.cross_entropy(out_edge_mask_seg, target.squeeze(1).long())
        loss_dice_edge_mask = dice_loss(F.softmax(out_edge_mask_seg, dim=1), target)

        edge_temp_mask = (edge != 0).float()
        if edge_temp_mask.sum() == 0:
            weight_matrix = torch.ones_like(edge_temp_mask) * 1.0
        else:
            pos_sum = (edge_temp_mask != 0).sum()
            neg_sum = (edge_temp_mask == 0).sum()
            pos_matrix = (neg_sum / pos_sum * 0.5) * edge_temp_mask
            neg_matrix = 1.0 * (1 - edge_temp_mask)
            weight_matrix = pos_matrix + neg_matrix
        loss_edge = (weight_matrix * torch.square(out_edge_reg - edge)).mean()
        # print('edge re-weight:', weight_matrix.max().item(), weight_matrix.min().item())

        #####  skeleton segmentor loss
        loss_ce_skele_mask = F.cross_entropy(out_skele_mask_seg, target.squeeze(1).long())
        loss_dice_skele_mask = dice_loss(F.softmax(out_skele_mask_seg, dim=1), target)   # notice

        skele_xyz = torch.cat((skele_x, skele_y, skele_z), dim=1)
        loss_flux = F.mse_loss(out_flux_reg, skele_xyz)
        skele_square = skele_xyz[:, 0, ...] ** 2 + skele_xyz[:, 1, ...] ** 2 + skele_xyz[:, 2, ...] ** 2
        out_flux_square = out_flux_reg[:, 0, ...] ** 2 + out_flux_reg[:, 1, ...] ** 2 + out_flux_reg[:, 2, ...] ** 2
        loss_flux_square = F.mse_loss(out_flux_square, skele_square)
        loss_flux = loss_flux + loss_flux_square


        loss = (loss_dice_skele_mask + loss_ce_skele_mask) + \
               (loss_dice_edge_mask + loss_ce_edge_mask) + \
               0.8 * loss_flux + \
               0.8 * loss_edge

        loss.backward()
        optimizer.step()

        dice_loss_edge = dice_score_metric(out_edge_mask_seg, target)
        dice_loss_skele = dice_score_metric(out_skele_mask_seg, target)
        pred = torch.argmax(out_edge_mask_seg, dim=1).unsqueeze(1)
        correct = pred.eq(target.data).cpu().sum()
        acc = correct / target.numel()

        correct_pos = torch.logical_and((pred == 1), (target == 1)).sum().item()
        sen_pos = round(correct_pos / ((target == 1).sum().item() + 0.0001), 3)

        correct_neg = torch.logical_and((pred == 0), (target == 0)).sum().item()
        sen_neg = round(correct_neg / ((target == 0).sum().item() + 0.0001), 3)

        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Base Train, Epoch: {:.2f} [{}/{} ({:.0f}%)]\t'
              'Loss: {:.5f}\t'
              'L_Dice_skele: {:.5f}\tL_CE_skele: {:.5f}\t L_flux_reg: {:.5f}\t'
              'L_Dice_edge: {:.5f}\tL_CE_edge: {:.5f}\t L_edge_seg: {:.5f}\t'
              'Acc: {:.3f}\tSen_pos: {:.3f}\tSen_neg: {:.3f}\tDice: {:.5f}\tDice_skele: {:.5f}'.format(
               partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
               loss.data,
               loss_dice_skele_mask.data, loss_ce_skele_mask.data, loss_flux.data,
               loss_dice_edge_mask.data, loss_ce_edge_mask.data, loss_edge.data,
               acc, sen_pos, sen_neg, dice_loss_edge, dice_loss_skele))

        trainF.write('{},{},{},{},{},{},{},{},{}\n'.format(partialEpoch, loss.data, loss_dice_skele_mask.data,
                                                           loss_dice_edge_mask.data, acc, sen_pos, sen_neg,
                                                           dice_loss_edge, dice_loss_skele))
        trainF.flush()

        wandb.log({
            "Train Dice Skele Mask Loss": loss_dice_skele_mask.item(),
            "Train Dice Edge Mask Loss": loss_dice_edge_mask.item(),
            "Train Skele Loss": loss_flux.item(),
            "Train Edge Loss": loss_edge.item(),
        })


def test_base(epoch, model, testLoader, testF, train_segmentor='base'):
    model.eval()
    test_loss = 0
    edge_error = 0
    skele_error = 0
    dice_score_edge = 0
    dice_score_skele = 0
    acc_all = 0
    sen_pos_all = 0
    sen_neg_all = 0
    with torch.no_grad():
        for data, target, edge, skele_x, skele_y, skele_z in testLoader:
            data, target, edge, skele_x, skele_y, skele_z = Variable(data.cuda()), Variable(target.cuda()), Variable(
                edge.cuda()), Variable(skele_x.cuda()), Variable(skele_y.cuda()), Variable(skele_z.cuda())

            out_flux_reg, out_skele_mask_seg, out_edge_reg, out_edge_mask_seg = model(data, train_segmentor)

            loss_dice_skele = dice_loss(F.softmax(out_skele_mask_seg, dim=1), target).data
            loss_dice_edge = dice_loss(F.softmax(out_edge_mask_seg, dim=1), target).data
            test_loss += loss_dice_skele + loss_dice_edge

            loss_edge = F.mse_loss(out_edge_reg, edge)
            edge_error += loss_edge

            skele_xyz = torch.cat((skele_x, skele_y, skele_z), dim=1)
            loss_flux = F.mse_loss(out_flux_reg, skele_xyz)
            skele_error += loss_flux

            dice_score_edge += dice_score_metric(out_edge_mask_seg, target)
            dice_score_skele += dice_score_metric(out_skele_mask_seg, target)

            pred = torch.argmax(out_edge_mask_seg, dim=1).unsqueeze(1)
            correct = pred.eq(target.data).cpu().sum()
            acc_all += correct / target.numel()

            correct_pos = torch.logical_and((pred == 1), (target == 1)).sum().item()
            sen_pos = round(correct_pos / ((target == 1).sum().item() + 0.0001), 3)
            sen_pos_all += sen_pos

            correct_neg = torch.logical_and((pred == 0), (target == 0)).sum().item()
            sen_neg = round(correct_neg / ((target == 0).sum().item() + 0.0001), 3)
            sen_neg_all += sen_neg

        test_loss /= len(testLoader)
        dice_score_edge /= len(testLoader)
        dice_score_skele /= len(testLoader)
        edge_error /= len(testLoader)
        skele_error /= len(testLoader)

        acc_all /= len(testLoader)
        sen_pos_all /= len(testLoader)
        sen_neg_all /= len(testLoader)

        print('\nTest online: Dice loss: {:.6f}, edge_error: {:.6f}, flux_error: {:.6f}, '
              '\tAcc:{:.3f}, Sen_pos:{:.3f}, Sen_neg:{:.3f}, '
              '\tDice_edge: {:.6f}, Dice_skele: {:.6f}\n'.format(
               test_loss, edge_error, skele_error,
               acc_all, sen_pos_all, sen_neg_all,
               dice_score_edge, dice_score_skele))

        testF.write(
            '{},{},{},{},{},{},{},{},{}\n'.format(epoch, test_loss, edge_error, skele_error, acc_all, sen_pos_all,
                                                  sen_neg_all, dice_score_edge, dice_score_skele))
        testF.flush()

        wandb.log({
            "Test Loss": test_loss,
            "Test Skele Error": skele_error,
            "Test Edge Error": edge_error,
            "Test Skele Dice Score": dice_score_skele,
            "Test Edge Dice Score": dice_score_edge,
        })

    return (dice_score_skele + dice_score_edge) * 0.5


def train_meta(epoch, model, trainLoader, optimizer, trainF, train_segmentor='meta', temperature=0.7):
    model.train()

    #################################################
    for name, para in model.module.named_parameters():
        if 'meta' in name:
            para.requires_grad = True
        else:
            para.requires_grad = False
    #################################################

    for batch_idx, (data, target, edge, skele_x, skele_y, skele_z) in enumerate(trainLoader):
        data, target, edge, skele_x, skele_y, skele_z = Variable(data.cuda()), Variable(target.cuda()), \
                                                        Variable(edge.cuda()), Variable(skele_x.cuda()), \
                                                        Variable(skele_y.cuda()), Variable(skele_z.cuda())
        optimizer.zero_grad()

        out_meta_mask_seg, out_edge, out_skele = model(data, train_segmentor)

        prob_edge = F.softmax(out_edge / temperature, dim=1)
        prob_skele = F.softmax(out_skele / temperature, dim=1)

        alpha = np.random.uniform(0.0, 1.0, 1)[0]
        soft_pseudo_target = alpha * prob_edge + (1 - alpha) * prob_skele
        loss_dice = dice_loss_PL(F.softmax(out_meta_mask_seg / temperature, dim=1), soft_pseudo_target)
        loss_ce = torch.mean(-(soft_pseudo_target * F.log_softmax(out_meta_mask_seg / temperature, dim=1)).sum(dim=1))
        loss_kl = F.kl_div(F.log_softmax(out_meta_mask_seg / temperature, dim=1), soft_pseudo_target, reduction='none').sum(dim=1).mean()

        loss = loss_ce + loss_dice

        loss.backward()
        optimizer.step()

        dice_score_meta = dice_score_metric(out_meta_mask_seg, target)

        pred = torch.argmax(out_meta_mask_seg, dim=1).unsqueeze(1)
        correct = pred.eq(target.data).cpu().sum()
        acc = correct / target.numel()

        correct_pos = torch.logical_and((pred == 1), (target == 1)).sum().item()
        sen_pos = round(correct_pos / ((target == 1).sum().item() + 0.0001), 3)

        correct_neg = torch.logical_and((pred == 0), (target == 0)).sum().item()
        sen_neg = round(correct_neg / ((target == 0).sum().item() + 0.0001), 3)

        partialEpoch = epoch + batch_idx / len(trainLoader) - 1

        print('Meta Train, Epoch: {:.2f}\tloss: {:.5f}\tLoss_dice: {:.5f}\tLoss_ce: {:.5f}\tAcc: {:.5f}\tSen: {:.5f}\tSpe: {:.5f}\tDice_score: {:.5f}'.
              format(partialEpoch, loss.item(), loss_dice.item(), loss_ce.item(), acc, sen_pos, sen_neg, dice_score_meta))

        trainF.write('{},{},{},{},{},{},{},{}\n'.
                     format(partialEpoch, loss.item(), loss_dice.item(), loss_ce.item(), acc, sen_pos, sen_neg, dice_score_meta))

        wandb.log({
            "Train Meta Mask Loss": loss.item(),
            "Train Meta CE Loss": loss_ce.item(),
            "Train Meta Dice Loss": loss_dice.item(),
        })

        trainF.flush()


def test_meta(epoch, model, testLoader, testF, train_segmentor='meta'):
    model.eval()
    test_loss = 0
    dice_score_meta = 0
    acc_all = 0
    sen_pos_all = 0
    sen_neg_all = 0
    with torch.no_grad():
        for data, target, edge, skele_x, skele_y, skele_z in testLoader:
            data, target, edge, skele_x, skele_y, skele_z = Variable(data.cuda()), Variable(target.cuda()), Variable(
                edge.cuda()), Variable(skele_x.cuda()), Variable(skele_y.cuda()), Variable(skele_z.cuda())

            out_meta_mask_seg, prob_edge, prob_skele = model(data, train_segmentor)

            loss_dice_meta = dice_loss(F.softmax(out_meta_mask_seg, dim=1), target).data
            test_loss += loss_dice_meta

            dice_score_meta += dice_score_metric(out_meta_mask_seg, target)

            pred = torch.argmax(out_meta_mask_seg, dim=1).unsqueeze(1)
            correct = pred.eq(target.data).cpu().sum()
            acc_all += correct / target.numel()

            correct_pos = torch.logical_and((pred == 1), (target == 1)).sum().item()
            sen_pos = round(correct_pos / ((target == 1).sum().item() + 0.0001), 3)
            sen_pos_all += sen_pos

            correct_neg = torch.logical_and((pred == 0), (target == 0)).sum().item()
            sen_neg = round(correct_neg / ((target == 0).sum().item() + 0.0001), 3)
            sen_neg_all += sen_neg

        test_loss /= len(testLoader)
        dice_score_meta /= len(testLoader)

        acc_all /= len(testLoader)
        sen_pos_all /= len(testLoader)
        sen_neg_all /= len(testLoader)
        print('\nTest online: Dice loss: {:.5f}, Acc:{:.3f}, Sen:{:.3f}, Spe:{:.3f}, Dice_meta: {:.5f}\n'.format(
             test_loss, acc_all, sen_pos_all, sen_neg_all, dice_score_meta))

        testF.write(
            '{},{},{},{},{},{}\n'.format(epoch, test_loss, acc_all, sen_pos_all, sen_neg_all, dice_score_meta))


        wandb.log({
            "Test Meta Loss": test_loss,
            "Test Meta Dice Score": dice_score_meta,
        })


        testF.flush()

    return dice_score_meta


if __name__ == '__main__':
    main()
