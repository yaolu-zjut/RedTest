import datetime
import os
import time
import torch
import tqdm
from nats_bench import create
from CKA import linear_CKA, unbias_CKA
from args import args
from utils.logging import AverageMeter, ProgressMeter
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
# from utils.Get_model import get_model, resnet20, get_NAS_model
from utils.Get_dataset import get_dataset
from torchvision.models.googlenet import googlenet
from utils.design_for_hook import *
from utils.get_new_model import get_NAS_model, get_model
from utils.utils import set_gpu, get_logger, accuracy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

'''
# setup up:
python calculating_MSRS.py --gpu 0 --arch resnet20 --set cifar10 --num_classes 10 --batch_size 256 --pretrained  --evaluate
'''

# Select the layer to be pruned
def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set)):
        os.makedirs('experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set), exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set) + '/logger' + now + '.log')
    logger.info(args)

    if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        api = create(None, 'tss', fast_mode=True, verbose=True)
        info = api.get_cost_info(int(args.arch), args.set)
        logger.info('flops:{}--params:{}--latency:{}'.format(info['flops'], info['params'], info['latency']))
        model = get_NAS_model(args, api)
    else:
        model = get_model(args)
    logger.info(model)
    model = set_gpu(args, model)
    CKA_whole_list = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    batch_count = 0
    if args.evaluate:
        if args.set in ['cifar10', 'cifar100']:
            if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                acc1, acc5 = NATS_Bench_validate(data.val_loader, model, criterion, args)
            else:
                acc1, acc5 = validate(data.val_loader, model, criterion, args)
        else:
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

        logger.info(acc1)

    inter_feature = []
    def hook(module, input, output):
        print(output.shape)
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            batch_count += 1
            if args.set == 'imagenet_dali':
                images = data[0]["data"].cuda(non_blocking=True)
            else:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'resnet402', 'resnet102']:
                handle_list = get_inner_feature_for_smallresnet(model, hook, args.arch)
            elif args.arch in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
                handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
            else:
                if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    handle_list = get_inner_feature_for_vgg(model, hook, 'NATS-Bench_tss')
                else:
                    handle_list = get_inner_feature_for_vgg(model, hook, args.arch)

            model(images)

            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    # inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1).cuda(device=args.gpu)  # no operation
                    inter_feature[m] = inter_feature[m].mean(dim=(2, 3)).cuda(device=args.gpu)  # mean

            CKA_list = CKA_for_layer_pruning(inter_feature)
            logger.info(CKA_list)
            CKA_whole_list.append(CKA_list)
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 10:  # multiple experiments
                break

    _, mean, _ = plot_mean_std_picture(CKA_whole_list)
    logger.info('CKA: {}'.format(np.array(mean)))


def CKA_for_layer_pruning(inter_feature):
    layer_num = len(inter_feature)
    CKA_list = []
    for ll in range(layer_num - 1):
        CKA_list.append(unbias_CKA(inter_feature[ll], inter_feature[ll+1]))

    return CKA_list


def plot_mean_std_picture(total_modularity_list):
    mean = []
    std = []
    x = []
    modular = []
    for kk in range(len(total_modularity_list[00])):  # layer numbers
        for jj in range(len(total_modularity_list)):  # repeat times
            modular.append(total_modularity_list[jj][kk])

        # print(modular)
        mean.append(torch.mean(torch.tensor(modular)))
        std.append(torch.std(torch.tensor(modular)))
        modular = []
        x.append(kk+1)

    return x, mean, std


def NATS_Bench_validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            _, output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return top1.avg, top5.avg

if __name__ == "__main__":
    main()
