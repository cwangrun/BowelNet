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
from torch.utils.data import DataLoader
from datasets.BowelDatasetCoarseSeg import BowelCoarseSeg
from tools.loss import *
import os, math
import shutil
import loc_model
import wandb


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'   # "1, 2"
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
    parser.add_argument('--batch_size', type=int, default=8)    # 8 (2 GPU)
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--nEpochs', type=int, default=1500, help='total training epoch')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--eval_interval', default=5, type=int, help='evaluate interval on validation set')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--deterministic', type=bool, default=True)
    args = parser.parse_args()


    args.save_dir = 'exp/BowelLocNet.{}'.format(datestr())
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)



    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'infer.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'crop_ROI.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'loc_model.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'tools/loss.py'), dst=args.save_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'datasets/BowelDatasetCoarseSeg.py'), dst=args.save_dir)


    # if args.seed is not None:
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     torch.backends.cudnn.deterministic = args.deterministic


    print("build Bowel Localisation Network")
    model = loc_model.BowelLocNet(elu=False)
    model.apply(weights_init)


    # model.load_state_dict(torch.load("./exp/BowelLocNet.20230208_1536/partial_5C_dict_1300.pth"))


    model = model.cuda()
    model = nn.parallel.DataParallel(model)

    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))



    # WandB – Initialize a new run
    wandb.init(project='BowelNet', mode='disabled')     # mode='disabled'
    wandb.run.name = 'BowelLoc_' + wandb.run.id



    print("loading fully_labeled dataset")
    fully_labeled_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Fully_labeled_5C'
    trainSet_fully_labeled = BowelCoarseSeg(fully_labeled_dir, mode="train", transform=True, dataset_name="fully_labeled", save_dir=args.save_dir)
    trainLoader_fully_labeled = DataLoader(trainSet_fully_labeled, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=False)
    testSet_fully_labeled = BowelCoarseSeg(fully_labeled_dir, mode="test", transform=False, dataset_name="fully_labeled", save_dir=args.save_dir)
    testLoader_fully_labeled = DataLoader(testSet_fully_labeled, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=False)

    print("loading smallbowel dataset")
    smallbowel_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Smallbowel'
    trainSet_small = BowelCoarseSeg(smallbowel_dir, mode="train", transform=True, dataset_name="smallbowel", save_dir=args.save_dir)
    trainLoader_small = DataLoader(trainSet_small, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    testSet_small = BowelCoarseSeg(smallbowel_dir, mode="test", transform=False, dataset_name="smallbowel", save_dir=args.save_dir)
    testLoader_small = DataLoader(testSet_small, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    print("loading colon_sigmoid dataset")
    colon_sigmoid_dir = '/mnt/c/chong/data/Bowel/crop_downsample/Colon_Sigmoid/colon_sigmoid'
    trainSet_colon_sigmoid = BowelCoarseSeg(colon_sigmoid_dir, mode="train", transform=True, dataset_name="colon_sigmoid", save_dir=args.save_dir)
    trainLoader_colon_sigmoid = DataLoader(trainSet_colon_sigmoid, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    testSet_colon_sigmoid = BowelCoarseSeg(colon_sigmoid_dir, mode="test", transform=False, dataset_name="colon_sigmoid", save_dir=args.save_dir)
    testLoader_colon_sigmoid = DataLoader(testSet_colon_sigmoid, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)   # 0.90


    best_dice = 0.0
    trainF = open(os.path.join(args.save_dir, 'train.csv'), 'w')
    testF = open(os.path.join(args.save_dir, 'test.csv'), 'w')
    for epoch in range(0, args.nEpochs + 1):
        if epoch < 0:   # 200
            train_loc(epoch, model, trainLoader_fully_labeled, optimizer, trainF, data_type='fully_labeled')
            # train_loc(epoch, model, trainLoader_small, optimizer, trainF, data_type='smallbowel')
            # train_loc(epoch, model, trainLoader_colon_sigmoid, optimizer, trainF, data_type='colon_sigmoid')
        else:
            if epoch % 3 == 0:
                train_loc(epoch, model, trainLoader_fully_labeled, optimizer, trainF, data_type='fully_labeled')
            elif epoch % 3 == 1:
                train_loc(epoch, model, trainLoader_small, optimizer, trainF, data_type='smallbowel')
            else:
                train_loc(epoch, model, trainLoader_colon_sigmoid, optimizer, trainF, data_type='colon_sigmoid')

        scheduler.step()
        wandb.log({
            "LR": optimizer.param_groups[0]['lr'],
        })

        if epoch % args.eval_interval == 0:  # 5
            dice_fully_labeled = test_loc(epoch, model, testLoader_fully_labeled, testF, data_type='fully_labeled')
            dice_small = test_loc(epoch, model, testLoader_small, testF, data_type='smallbowel')
            dice_colon_sigmoid = test_loc(epoch, model, testLoader_colon_sigmoid, testF, data_type='colon_sigmoid')
            dice_mean = (np.mean(dice_fully_labeled) + np.mean(dice_small) + np.mean(dice_colon_sigmoid)) / 3.0

            wandb.log({
                "Test Mean Dice Score": dice_mean,
            })

            is_best = False
            if dice_mean > best_dice:
                is_best = True
                best_dice = dice_mean
            save_checkpoint({'epoch': epoch, 'state_dict': model.module.state_dict(), 'best_acc': best_dice},
                              is_best, args.save_dir, "partial_5C")
            torch.save(model.module.state_dict(), args.save_dir + '/partial_5C_dict_' + str(epoch) + '.pth')

    trainF.close()
    testF.close()


def train_loc(epoch, model, trainLoader, optimizer, trainF, data_type):

    assert data_type in ['fully_labeled', 'smallbowel', 'colon_sigmoid']

    model.train()
    nProcessed = 0
    nTrain = len(trainLoader)

    for batch_idx, (data, target, image_name) in enumerate(trainLoader):

        data, target = Variable(data.cuda()), Variable(target.cuda())

        optimizer.zero_grad()

        out = model(data)
        out = torch.clamp(out, min=1e-10, max=1)    # prevent overflow

        # {1: 'rectum', 2: 'sigmoid', 3: 'colon', 4: 'small', 5: 'duodenum'}
        if data_type == 'fully_labeled':
            output_p = out
            target_ce = target.long()
            target_dice = torch.cat((target_ce == 0, target_ce == 1, target_ce == 2,
                                     target_ce == 3, target_ce == 4, target_ce == 5), dim=1).long()
            loss_entropy = torch.tensor(0.0).cuda()

        if data_type == 'smallbowel':
            output_p1 = out[:, 4, :, :, :]        # smallbowel
            output_p0 = out[:, 0, :, :, :] + \
                        out[:, 1, :, :, :] + \
                        out[:, 2, :, :, :] + \
                        out[:, 3, :, :, :] + \
                        out[:, 5, :, :, :]
            output_p = torch.stack((output_p0, output_p1), dim=1)
            target_ce = (target == 4).long()
            target_dice = torch.cat((target_ce == 0, target_ce == 1), dim=1).long()
            prob_neg = torch.stack((out[:, 0],
                                    out[:, 1],
                                    out[:, 2],
                                    out[:, 3],
                                    out[:, 5]), dim=1)
            entropy = (-prob_neg * torch.log(prob_neg)).sum(dim=1)
            neg_mask = (target_ce == 0).squeeze(1)
            loss_entropy = (entropy * neg_mask).sum() / ((neg_mask).sum() + 1e-7)

        if data_type == 'colon_sigmoid':
            output_p1 = out[:, 2, :, :, :]       # sigmoid
            output_p2 = out[:, 3, :, :, :]       # colon
            output_p0 = out[:, 0, :, :, :] + \
                        out[:, 1, :, :, :] + \
                        out[:, 4, :, :, :] + \
                        out[:, 5, :, :, :]
            output_p = torch.stack((output_p0, output_p1, output_p2), dim=1)
            target_ce = torch.zeros(target.shape).long().cuda()
            target_ce[target == 2] = 1
            target_ce[target == 3] = 2
            target_dice = torch.cat((target_ce == 0, target_ce == 1, target_ce == 2), dim=1).long()
            prob_neg = torch.stack((out[:, 0],
                                    out[:, 1],
                                    out[:, 4],
                                    out[:, 5]), dim=1)
            entropy = (-prob_neg * torch.log(prob_neg)).sum(dim=1)
            neg_mask = (target_ce == 0).squeeze(1)
            loss_entropy = (entropy * neg_mask).sum() / ((neg_mask).sum() + 1e-7)

        loss_ce = F.nll_loss(output_p.log(), target_ce.squeeze(1))
        loss_dice = dice_loss_PL(output_p, target_dice)
        loss = loss_dice + loss_ce + loss_entropy

        loss.backward()
        optimizer.step()

        dice_score = dice_score_partial(out, target, data_type)

        pred = torch.argmax(output_p, dim=1, keepdim=True)
        correct = pred.eq(target_ce.data).cpu().sum()
        acc = correct / target_ce.numel()

        correct_pos = torch.logical_and((pred != 0), (target_ce != 0)).sum().item()
        sen_pos = round(correct_pos / ((target_ce != 0).sum().item() + 0.0001), 3)

        correct_neg = torch.logical_and((pred == 0), (target_ce == 0)).sum().item()
        sen_neg = round(correct_neg / ((target_ce == 0).sum().item() + 0.0001), 3)

        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / nTrain
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tData_type: {}\tLoss: {:.4f}\tLoss_Dice: {:.4f}\tLoss_CE: {:.4f} '
              '\tLoss_Ent: {:.4f}'
              '\tAcc: {:.3f}\tSen_pos: {:.3f}\tSen_neg: {:.3f}\tDice: {}'.format(
            partialEpoch, nProcessed, len(trainLoader.dataset), 100. * batch_idx / nTrain, data_type,
            loss.data, loss_dice.data, loss_ce.data, loss_entropy.data, acc, sen_pos, sen_neg, dice_score))

        wandb.log({
            "Train Loss": loss,
            "Train CE Loss": loss_ce,
            "Train Dice Loss": loss_dice,
            "Train Entropy Loss": loss_entropy,
        })

        trainF.write(
            '{},{},{},{},{},{},{},{},{}, {}\n'.format(partialEpoch, data_type, loss.data, loss_dice.data, loss_ce.data, loss_entropy.data, acc, sen_pos, sen_neg, dice_score))
        trainF.flush()


def test_loc(epoch, model, testLoader, testF, data_type):

    assert data_type in ['fully_labeled', 'smallbowel', 'colon_sigmoid']

    model.eval()
    test_loss = 0
    dice_score = 0
    acc_all = 0
    sen_pos_all = 0
    sen_neg_all = 0
    with torch.no_grad():
        for data, target, image_name in testLoader:
            data, target = Variable(data.cuda()), Variable(target.cuda())

            out = model(data)

            # {1: 'rectum', 2: 'sigmoid', 3: 'colon', 4: 'small', 5: 'duodenum'}

            if data_type == 'fully_labeled':
                output_p = out
                target_ce = target.long()
                target_dice = torch.cat((target_ce == 0, target_ce == 1, target_ce == 2,
                                         target_ce == 3, target_ce == 4, target_ce == 5), dim=1).long()

            if data_type == 'smallbowel':
                output_p1 = out[:, 4, :, :, :]  # smallbowel
                output_p0 = out[:, 0, :, :, :] + \
                            out[:, 1, :, :, :] + \
                            out[:, 2, :, :, :] + \
                            out[:, 3, :, :, :] + \
                            out[:, 5, :, :, :]
                output_p = torch.stack((output_p0, output_p1), dim=1)
                target_ce = (target == 4).long()
                target_dice = torch.cat((target_ce == 0, target_ce == 1), dim=1).long()

            if data_type == 'colon_sigmoid':
                output_p1 = out[:, 2, :, :, :]  # sigmoid
                output_p2 = out[:, 3, :, :, :]  # colon
                output_p0 = out[:, 0, :, :, :] + \
                            out[:, 1, :, :, :] + \
                            out[:, 4, :, :, :] + \
                            out[:, 5, :, :, :]
                output_p = torch.stack((output_p0, output_p1, output_p2), dim=1)
                target_ce = torch.zeros(target.shape).long().cuda()
                target_ce[target == 2] = 1
                target_ce[target == 3] = 2
                target_dice = torch.cat((target_ce == 0, target_ce == 1, target_ce == 2), dim=1).long()


            loss_dice = dice_loss_PL(output_p, target_dice)
            test_loss += loss_dice

            dice_score += np.array(dice_score_partial(out, target, data_type))

            pred = torch.argmax(output_p, dim=1, keepdim=True)
            correct = pred.eq(target_ce.data).cpu().sum()
            acc_all += correct / target_ce.numel()

            correct_pos = torch.logical_and((pred != 0), (target_ce != 0)).sum().item()
            sen_pos = round(correct_pos / ((target_ce != 0).sum().item() + 0.0001), 3)
            sen_pos_all += sen_pos

            correct_neg = torch.logical_and((pred == 0), (target_ce == 0)).sum().item()
            sen_neg = round(correct_neg / ((target_ce == 0).sum().item() + 0.0001), 3)
            sen_neg_all += sen_neg

        test_loss /= len(testLoader)
        dice_score /= len(testLoader)
        acc_all /= len(testLoader)
        sen_pos_all /= len(testLoader)
        sen_neg_all /= len(testLoader)
        print('\nTest online: Data_type: {}, Dice loss: {:.4f}, Acc:{:.3f}, Sen_pos:{:.3f}, Sen_neg:{:.3f}, Dice: {}\n'.
            format(data_type, test_loss, acc_all, sen_pos_all, sen_neg_all, dice_score))

        testF.write('{},{},{},{},{},{},{}\n'.format(epoch, data_type, test_loss, acc_all, sen_pos_all, sen_neg_all, dice_score))
        testF.flush()

        wandb.log({
            "Test Loss": test_loss,
            "Test Dice Score": dice_score.mean(),
        })

    return dice_score


if __name__ == '__main__':
    main()
