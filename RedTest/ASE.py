import datetime
import os
import torch
import tqdm
from nats_bench import create
from CKA import linear_CKA, unbias_CKA
from NAS_eval import NATS_Bench_validate
from args import args
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
# from utils.Get_model import get_model, resnet20, get_NAS_model
from utils.Get_dataset import get_dataset
from utils.get_new_model import get_NAS_model, get_model
from torchvision.models.googlenet import googlenet
from utils.design_for_hook import *
from utils.utils import set_gpu, get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import pandas as pd

'''
# setup up:
python ASE.py --gpu 0 --arch resnet20 --set cifar10 --num_classes 10 --batch_size 256 --pretrained  --evaluate
'''

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('new_experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set)):
        os.makedirs('new_experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set), exist_ok=True)
    logger = get_logger('new_experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set) + '/logger' + now + '.log')
    logger.info(args)

    if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        api = create(None, 'tss', fast_mode=True, verbose=True)  # '/public/ly/.torch/NATS-tss-v1_0-3ffb9-full'
        info = api.get_cost_info(int(args.arch), args.set)
        logger.info('flops:{}--params:{}--latency:{}'.format(info['flops'], info['params'], info['latency']))
        model = get_NAS_model(args, api)
    else:
        model = get_model(args)
        # model = resnet20(num_classes=args.num_classes)
        # save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20.th', map_location='cuda:%d' % args.gpu)
        # ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
        # if args.pretrained:
        #     model.load_state_dict(ckpt)
    logger.info(model)
    model = set_gpu(args, model)
    MSRS_list = []
    CKA_matrix_list = []
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
                # target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'resnet56_CKA_c10_15', 'resnet1202_cka_15', 'resnet402', 'resnet102']:
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
                    # inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1).cuda(device=args.gpu)
                    inter_feature[m] = inter_feature[m].mean(dim=(2, 3)).cuda(device=args.gpu)

            CKA_matrix_for_visualization = CKA_heatmap(inter_feature)
            if args.arch in ['cvgg11_bn', 'cvgg13_bn', 'cvgg16_bn', 'cvgg19_bn', 'Ivgg11_bn', 'Ivgg13_bn', 'Ivgg16_bn', 'Ivgg19_bn', 'VGG19_Shallowing_c10', 'VGG19_modularity_c10', 'VGG19_CKA_c10']:
                threshold = 0.4
            else:
                threshold = 0.8

            logger.info('Threshold = {}'.format(threshold))
            MSRS = Tanh_MSRS(CKA_matrix_for_visualization, beta=100, threshold=threshold)
            logger.info('MSRS:{}'.format(MSRS))
            if torch.isnan(MSRS):
                print('This MSRS is nan, we ignore it!')
            else:
                MSRS_list.append(MSRS)
            CKA_matrix_list.append(CKA_matrix_for_visualization)
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 10:  # multiple experiments
                break

    logger.info('\r\n')
    logger.info('Average MSRS:{}'.format(sum(MSRS_list) / len(MSRS_list)))
    save_path = 'new_experiment/' + args.arch + '/' + 'CKA_%d_%s' % (args.batch_size, args.set) + '/'
    torch.save(CKA_matrix_list, save_path + 'CKA_matrix_for_visualization.pth')


def CKA_heatmap(inter_feature):
    layer_num = len(inter_feature)
    CKA_matrix = torch.zeros((layer_num, layer_num))
    for ll in range(layer_num):
        for jj in range(layer_num):
            if ll < jj:
                # CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = linear_CKA(inter_feature[ll], inter_feature[jj])
                CKA_matrix[ll, jj] = CKA_matrix[jj, ll] = unbias_CKA(inter_feature[ll], inter_feature[jj])

    CKA_matrix_for_visualization = CKA_matrix + torch.eye(layer_num)
    return CKA_matrix_for_visualization


def Tanh_MSRS(sim_matrix, beta=100, threshold=0.8):
    layer_num = sim_matrix.shape[0]
    sim_matrix = sim_matrix - torch.eye(layer_num)
    sim_matrix = beta * (sim_matrix - threshold)
    out = (torch.exp(sim_matrix) - torch.exp(-sim_matrix)) / (torch.exp(sim_matrix) + torch.exp(-sim_matrix))
    return torch.sum(( out + 1 ) / 2) / 2

if __name__ == "__main__":
    main()
