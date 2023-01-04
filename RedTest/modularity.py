import networkx as nx
import numpy as np
import datetime
import os
import torch
import tqdm
from nats_bench import create
from args import args
from draw_networkx import directly_return_undirected_weighted_network
from trainer.amp_trainer_dali import validate_ImageNet
from trainer.trainer import validate
from utils.calculate_similarity import calculate_cosine_similarity_matrix
from utils.design_for_hook import get_inner_feature_for_resnet, get_inner_feature_for_vgg, get_inner_feature_for_smallresnet
from utils.Get_dataset import get_dataset
from utils.get_new_model import get_NAS_model, get_model
from utils.utils import set_gpu, get_logger

'''
# setup up:
python modularity.py --gpu 0 --arch resnet56 --set cifar10 --num_classes 10 --batch_size 500 --pretrained
'''

def main():
    assert args.pretrained, 'this program needs pretrained model'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.isdir('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set):
        os.makedirs('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set, exist_ok=True)
    logger = get_logger('experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/logger' + now + '.log')
    logger.info(args)
    if isinstance(args.arch, int):
        api = create(None, 'tss', fast_mode=True, verbose=True)
        info = api.get_cost_info(args.arch, args.set)
        logger.info('flops:{}--params:{}--latency:{}'.format(info['flops'], info['params'], info['latency']))
        model = get_NAS_model(args, api)
    else:
        model = get_model(args)
    print(model)

    model = set_gpu(args, model)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    data = get_dataset(args)
    model.eval()
    batch_count = 0
    if args.evaluate:
        if args.set in ['cifar10', 'cifar100']:
            acc1, acc5 = validate(data.val_loader, model, criterion, args)
        else:
            acc1, acc5 = validate_ImageNet(data.val_loader, model, criterion, args)

        logger.info(acc1)

    inter_feature = []
    modularity_list = []
    total_modularity_list = []
    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            batch_count += 1
            if args.set == 'imagenet_dali':
                images = data[0]["data"].cuda(non_blocking=True)
                target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            else:
                images, target = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)

            print(target)
            if args.arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'resnet56_modularitylayerpruning']:
                handle_list = get_inner_feature_for_smallresnet(model, hook, args.arch)
            elif args.arch in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']:
                handle_list = get_inner_feature_for_resnet(model, hook, args.arch)
            else:
                if isinstance(args.arch, int):
                    handle_list = get_inner_feature_for_vgg(model, hook, 'NATS-Bench_tss')
                else:
                    handle_list = get_inner_feature_for_vgg(model, hook, args.arch)

            output = model(images)
            for m in range(len(inter_feature)):
                print('-'*50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1)

                similarity_matrix, edges_list = calculate_cosine_similarity_matrix(inter_feature[m], args.topk, eps=1e-8)
                undirected_weighted_network, undirected_adj = directly_return_undirected_weighted_network(edges_list)
                modularity = calculate_modularity(undirected_weighted_network, target, class_num=args.num_classes)
                modularity_list.append(modularity)

            logger.info(modularity_list)
            total_modularity_list.append(modularity_list)
            modularity_list = []
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 5:
                break

    torch.save(total_modularity_list, 'experiment/' + args.arch + '/' + 'Modularity_%s' % args.set + '/' + 'top%d_batch%d_num%d' % (args.topk, args.batch_size, batch_count))


def calculate_modularity(G, target, class_num=5):  # absolutely sure
    r'''

    Args:
        G: undirected weighted graph
        target: category of nodes, like [[1, 3], [0, 2, 4]] or [{1, 3}, {0, 2, 4}]
        class_num: the number of class images

    Returns: modularity

    '''
    node_category = []
    # deal with nodes' category
    for i in range(class_num):
        a = [j for j, x in enumerate(target) if x == i]
        node_category.append(a)
        # node 0 to batch_size, like
        # [[0, 4, 13, 19, 20, 29], [5, 6, 7, 12, 16, 17], [8, 9, 22, 23, 24, 27], [10, 11, 15, 18, 21, 25], [1, 2, 3, 14, 26, 28]]

    modularity = nx.algorithms.community.quality.modularity(G, node_category)
    return modularity


if __name__ == "__main__":
    main()
