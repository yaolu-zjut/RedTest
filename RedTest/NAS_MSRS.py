import datetime
import sys
import torch
import tqdm
import numpy as np
from calculating_MSRS import CKA_heatmap, Tanh_MSRS
from args import args
from nats_bench import create
from pprint import pprint
from xautodl.models import get_cell_based_tiny_net
from nats_bench.api_utils import time_string
from utils.Get_dataset import get_dataset
from utils.design_for_hook import get_inner_feature_for_vgg
from utils.utils import set_gpu, get_logger, Logger

def train_model():
    # python NAS_MSRS.py --gpu 2 --set cifar10 --batch_size 256 --num_classes 10
    # set: "cifar10", "cifar100", 'ImageNet16-120'
    # num_classes: 10, 100, 120
    sys.stdout = Logger('print process.log', sys.stdout)
    mode = ["01", "12", "90", "200"]
    info_dict = {'12-epoch-testing acc': [], '200-epoch-testing acc': [], 'flops': [], 'params': [], 'latency': [], 'MSRS':[], 'MSRS1': []}
    model_id = np.load('/public/ly/ASE/NATS-Bench-experiments/pretrained_model_id/model_id.npy')
    print('There are {} pretrained weights in this paper'.format(len(model_id)))

    api = create('/public/ly/.torch/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=True)
    print('{:} There are {:} architectures on the topology search space'.format(time_string(), len(api)))
    # query_best_performance(api)

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger = get_logger('/public/ly/ASE/NATS-Bench-experiments/' + '/logger' + now + '.log')
    logger.info(args)

    start = 0
    end = 1000
    # Since MSRS calculated by some models are NAN, we do not consider them.
    # In addition, there are invalid models in the model pool, which will cause the code to break.
    # So we execute the code multiple times. (start & end)
    logger.info('start:{}--end:{}'.format(start, end))
    for j in range(start, end):  # 13489
        id = int(model_id[j])
        info_dict = main(id, api, logger, info_dict)

    print(info_dict)
    torch.save(info_dict, '/public/ly/ASE/NATS-Bench-experiments/whole_statistical_data/data{}-{}-{}.pth'.format(start, end, args.set))
    return


def main(id, api, logger, info_dict):
    logger.info('###################### {}-th candidate architecture on {} ######################'.format(id, args.set))
    inter_feature = []

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())

    # Query the loss / accuracy / time for 12-th candidate architecture on CIFAR-10
    # info is a dict, where you can easily figure out the meaning by key
    info = api.get_more_info(id, args.set, hp='12')
    info_dict['12-epoch-testing acc'].append(info['test-accuracy'])
    logger.info('12-epoch-testing acc: {}'.format(info['test-accuracy']))
    info = api.get_more_info(id, args.set, hp='200')
    info_dict['200-epoch-testing acc'].append(info['test-accuracy'])
    logger.info('200-epoch-testing acc: {}'.format(info['test-accuracy']))

    # Query the flops, params, latency. info is a dict.
    info = api.get_cost_info(id, args.set)
    logger.info('flops:{}--params:{}--latency:{}'.format(info['flops'], info['params'], info['latency']))
    info_dict['flops'].append(info['flops'])
    info_dict['params'].append(info['params'])
    info_dict['latency'].append(info['latency'])

    config = api.get_net_config(id, args.set)
    network = get_cell_based_tiny_net(config)

    # Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
    params = api.get_net_param(id, args.set, None, hp='12')  # train the model by 12 epochs
    network.load_state_dict(next(iter(params.values())))

    model = set_gpu(args, network)
    data = get_dataset(args)
    model.eval()
    batch_count = 0
    MSRS_list = []
    CKA_matrix_list = []

    with torch.no_grad():
        for i, data in tqdm.tqdm(
                enumerate(data.val_loader), ascii=True, total=len(data.val_loader)
        ):
            batch_count += 1
            images = data[0].cuda(args.gpu, non_blocking=True)
            handle_list = get_inner_feature_for_vgg(model, hook, 'NATS-Bench_tss')
            model(images)

            for m in range(len(inter_feature)):
                print('-' * 50)
                print(m)
                if len(inter_feature[m].shape) != 2:
                    # inter_feature[m] = inter_feature[m].reshape(args.batch_size, -1).cuda(device=args.gpu)
                    inter_feature[m] = inter_feature[m].mean(dim=(2, 3)).cuda(device=args.gpu)

            CKA_matrix_for_visualization = CKA_heatmap(inter_feature)
            MSRS = Tanh_MSRS(CKA_matrix_for_visualization, beta=100, threshold=0.8)
            MSRS_list.append(MSRS)
            CKA_matrix_list.append(CKA_matrix_for_visualization)
            inter_feature = []
            for i in range(len(handle_list)):
                handle_list[i].remove()

            if batch_count == 10:  # multiple experiments
                break

    Average_MSRS = sum(MSRS_list) / batch_count
    logger.info('Average MSRS:{}'.format(Average_MSRS))
    info_dict['MSRS'].append(Average_MSRS)
    logger.info('\r\n')

    return info_dict

def query_best_performance(api):
    # query the largest model's performance
    largest_candidate_tss = '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|'
    arch_index = api.query_index_by_arch(largest_candidate_tss)
    print('The architecture-index for the largest model is {:}'.format(arch_index))
    datasets = ('cifar10', 'cifar100', 'ImageNet16-120')
    for dataset in datasets:
        print('Its performance on {:} with 12-epoch-training'.format(dataset))
        info = api.get_more_info(arch_index, dataset, hp='12', is_random=False)
        pprint(info)
        print('Its performance on {:} with 200-epoch-training'.format(dataset))
        info = api.get_more_info(arch_index, dataset, hp='200', is_random=False)
        pprint(info)


if __name__ == "__main__":
    train_model()