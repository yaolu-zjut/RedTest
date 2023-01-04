import sys
import heapq
import copy
from xautodl.models import get_cell_based_tiny_net
import numpy as np
from model.VGG_cifar import *
import torch
from torchvision.models import *
from model.diy_cvgg import *
from model.samll_resnet import *
from utils.finetune_layer import get_cresnet_layer_params, load_cresnet_layer_params, load_vgg_layer_params


def get_model(args):
    # Note that you can train your own models using train.py
    # We will expose the model files later
    print(f"=> Getting {args.arch}")
    if args.arch == 'ResNet18':
        model = resnet18(pretrained=True)
    elif args.arch == 'ResNet34':
        model = resnet34(pretrained=True)
    elif args.arch == 'ResNet50':
        model = resnet50(pretrained=True)
    elif args.arch == 'ResNet101':
        model = resnet101(pretrained=True)
    elif args.arch == 'ResNet152':
        model = resnet152(pretrained=True)
    elif args.arch == 'cvgg11_bn':
        model = cvgg11_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg11_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg11_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg13_bn':
        model = cvgg13_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg13_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg13_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg16_bn':
        model = cvgg16_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg16_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'cvgg19_bn':
        model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet20':
        model = resnet20(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet20/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet32':
        model = resnet32(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet32.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet32/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet44':
        model = resnet44(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet44.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet44/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56':
        model = resnet56(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110':
        model = resnet110(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            elif args.set == 'cifar100':
                ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet102':
        model = resnet102(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet102/cifar10/scores9328.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet402':
        model = resnet402(num_classes=args.num_classes)
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet402/cifar10/scores9416.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet1202':
        model = resnet1202(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet1202.th', map_location='cuda:%d' % args.gpu)
                ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
    elif args.arch == 'resnet1202_cka_15':
        model = resnet1202_cka_15(num_classes=10)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/ASE/pretrained_model/resnet1202_cka_15/cifar10/scores9347.pt', map_location='cuda:%d' % args.gpu)
        if args.finetune:
            x = np.load('/public/ly/ASE/pretrained_model/resnet1202/resnet1202_pruning.npy').tolist()
            num = 15  # the number of remaining layer
            min_number = heapq.nsmallest(num, x)  # nsmallest: for smallest
            min_index = []
            copy_x = copy.deepcopy(x)
            for t in min_number:
                index = copy_x.index(t)
                min_index.append(index)
                copy_x[index] = 0

            remain_list = [[], [], []]
            for i in min_index:
                if i <= 200:
                    remain_list[0].append(i)
                elif i <= 400:
                    remain_list[1].append(i - 200)
                else:
                    remain_list[2].append(i - 400)

            remain_list[0].sort()
            remain_list[1].sort()
            remain_list[2].sort()

            orginal_model = resnet1202(num_classes=10)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet1202.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=200)
            print('Load pretrained weights from the original model')

    elif args.arch == 'Ivgg11_bn':
        model = vgg11_bn(pretrained=True)
    elif args.arch == 'Ivgg13_bn':
        model = vgg13_bn(pretrained=True)
    elif args.arch == 'Ivgg16_bn':
        model = vgg16_bn(pretrained=True)
    elif args.arch == 'Ivgg19_bn':
        model = vgg19_bn(pretrained=True)
    elif args.arch == 'resnet56_Shallowing':  # used OK 8.11
        model = resnet56_Shallowing(num_classes=args.num_classes)
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/ASE/pretrained_model/resnet56_Shallowing/cifar10/lp_scores.pt', map_location='cuda:%d' % args.gpu)
        if args.finetune:
            orginal_model = resnet56(num_classes=10)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 3, 6, 8], [0, 1, 3, 5, 7], [0, 1, 2, 3, 5]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
    elif args.arch == 'resnet56_Shallowing_c100':
        model = resnet56_Shallowing_c100(num_classes=100)
        if args.pretrained:
            if args.set == 'cifar100':
                ckpt = torch.load('/public/ly/ASE/pretrained_model/resnet56_Shallowing_c100/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56_modularity_c100':
        model = resnet56_modularity_c100(num_classes=100)
        if args.finetune:
            orginal_model = resnet56(num_classes=100)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[7], [0, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/pretrained_model/resnet56_modularity_c100/cifar100/scores6812.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56_modularity_c10':
        model = resnet56_modularity_c10(num_classes=10)
        if args.finetune:
            orginal_model = resnet56(num_classes=10)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[7, 8], [0, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/pretrained_model/resnet56_modularity_c10_rebuttal/cifar10/scores9338.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'VGG19_modularity_c10':
        model = cvgg19_modularity_c10(num_classes=10)
        if args.finetune:
            orginal_model = cvgg19_bn(num_classes=10, batch_norm=True)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_conv_list = [[0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49], [1, 4, 6]]
            pruned_conv_list = [[(0, 0), (4, 2), (8, 4), (11, 5), (14, 6), (17, 7), (21, 8), (24, 9), (27, 10), (31, 11)],
                                [(1, 0), (4, 1), (6, 2)]]  # match the orginal_conv_list and pruned model
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/ASE/new_pretrained_model/VGG19_modularity_c10/cifar10/scores9277.pt')
    elif args.arch == 'VGG19_CKA_c10':  # used OK 8.11
        # [0.81111956 0.5691167  0.7536302  0.7598313  0.84407616 0.8530037
        #  0.7927989  0.79135907 0.77015305 0.88825816 0.77569777 0.7998207
        #  0.91992533 0.86013335 0.9487301 ]
        model = VGG19_CKA_c10(num_classes=10)
        if args.finetune:
            orginal_model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_conv_list = [[0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49], [1, 4, 6]]
            pruned_conv_list = [[(0, 0), (4, 2), (7, 3), (11, 4), (14, 7), (18, 8), (21, 9), (25, 11)],
                                [(1, 0), (4, 1), (6, 2)]]  # match the orginal_conv_list and pruned model
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            if args.set == 'cifar10':
                ckpt = torch.load('/public/ly/ASE/new_pretrained_model/VGG19_CKA_c10/cifar10/scores9322.pt')
    elif args.arch == 'VGG19_Shallowing_c10':  # used OK 8.11
        model = VGG19_Shallowing_c10(num_classes=10)
        if args.finetune:
            orginal_model = cvgg19_bn(num_classes=args.num_classes, batch_norm=True)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/cvgg19_bn/cifar10/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_conv_list = [[0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49], [1, 4, 6]]
            pruned_conv_list = [[(0, 0), (3, 1), (7, 2), (11, 4), (14, 5), (17, 6), (20, 7), (24, 8), (28, 9)],
                                [(1, 0), (4, 1), (6, 2)]]  # match the orginal_conv_list and pruned model
            model = load_vgg_layer_params(orginal_model, orginal_conv_list, model, pruned_conv_list)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/pretrained_model/VGG19_Shallowing_c10/cifar10/lp_scores.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56_CKA_c10_14':  # used OK 8.10
        # [0.8040034 0.9623917 0.9816055 0.9818996 0.9646843 0.9950012 0.96210957 0.95488405 0.9958612
        # 0.9714489 0.99030924 0.9881511 0.99465436 0.99431086 0.9923767 0.9919683 0.9893893 0.9911849
        # 0.9701058 0.9823101 0.9843625 0.98163974 0.966809 0.93328685 0.9032259 0.8794835 0.8777317]
        model = resnet56_CKA_c10_14(num_classes=10)
        if args.finetune:
            orginal_model = resnet56(num_classes=10)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 4, 6, 7], [0], [0, 4, 5, 6, 7, 8]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet56_CKA_c10_14/cifar10/scores9337.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110_CKA_c10_15':  # used
    #     [0.94884205, 0.9770746, 0.95801353, 0.9939505, 0.98726845, 0.9893962, 0.98875153, 0.99533117, 0.997571,
    #      0.9886915, 0.97693574, 0.99091685, 0.9852234, 0.9873949, 0.9911769, 0.9832204, 0.9842418, 0.99356234,
    #      0.9645895, 0.99829596, 0.9973879, 0.99652565, 0.9983717, 0.9946502, 0.9971191, 0.9979045, 0.9978746,
    #      0.99804276, 0.9967415, 0.9954311, 0.99763125, 0.9974578, 0.996971, 0.99582374, 0.99504054, 0.9961294,
    #      0.982186, 0.9923633, 0.99419564, 0.99087715, 0.9947198, 0.99275845, 0.9914743, 0.99103945, 0.9932662,
    #      0.98934555, 0.98938334, 0.99056274, 0.98145306, 0.9834711, 0.965495, 0.9723147, 0.9525223, 0.89880216]
        model = resnet110_CKA_c10_15(num_classes=10)
        if args.finetune:
            orginal_model = resnet110(num_classes=10)
            save = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110.th', map_location='cuda:%d' % args.gpu)
            ckpt = {k.replace('module.', ''): v for k, v in save['state_dict'].items()}
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 1, 2, 10, 15, 16], [0], [0, 12, 13, 14, 15, 16, 17]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=18)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet110_CKA_c10_15/cifar10/scores935.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet56_CKA_c100_15':  # used
        model = resnet56_CKA_c100_15(num_classes=100)
        # [0.89893186, 0.98167545, 0.8873335, 0.99292386, 0.99385583, 0.9889188, 0.992875, 0.9876197, 0.9502824,
        #  0.9415885, 0.97993964, 0.97814405, 0.98698264, 0.9881584, 0.9923223, 0.9799606, 0.98457384, 0.97167224,
        #  0.9489094, 0.9498153, 0.91059005, 0.8620137, 0.8641551, 0.9412409, 0.96756804, 0.9870943, 0.9951965]
        if args.finetune:
            orginal_model = resnet56(num_classes=100)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet56/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 2, 8], [0, 1, 2, 8], [0, 1, 2, 3, 4, 5, 6]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=9)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet56_CKA_c100_15/cifar100/scores7061.pt', map_location='cuda:%d' % args.gpu)
    elif args.arch == 'resnet110_CKA_c100_16':  # used
        model = resnet110_CKA_c100_16(num_classes=100)
        # [0.93557775, 0.9792716, 0.95650053, 0.9941395, 0.9946949, 0.9853319, 0.99746835, 0.99607337, 0.99851835,
        #  0.9989565, 0.9962349, 0.9913951, 0.9932219, 0.9925826, 0.98335695, 0.99722975, 0.9600028, 0.99303514,
        #  0.9602537, 0.9886389, 0.99775743, 0.96976125, 0.991374, 0.9979828, 0.9915552, 0.9974679, 0.99442357,
        #  0.99303114, 0.99541825, 0.9949991, 0.99455535, 0.9932181, 0.995406, 0.9787676, 0.98938715, 0.9853932,
        #  0.9674503, 0.9751221, 0.9812791, 0.98032, 0.9749913, 0.9691599, 0.9447282, 0.9426031, 0.9150691,
        #  0.8937991, 0.9281022, 0.9595434, 0.97717637, 0.98760176, 0.99356556, 0.9965979, 0.9981247, 0.99893445]
        if args.finetune:
            orginal_model = resnet110(num_classes=100)
            ckpt = torch.load('/public/ly/Dynamic_Graph_Construction/pretrained_model/resnet110/cifar100/scores.pt', map_location='cuda:%d' % args.gpu)
            orginal_model.load_state_dict(ckpt)
            orginal_state_list = get_cresnet_layer_params(orginal_model)
            remain_list = [[0, 2, 16], [0, 3], [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]]
            model = load_cresnet_layer_params(orginal_state_list, model, remain_list, num_of_block=18)
            print('Load pretrained weights from the original model')
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/new_pretrained_model/resnet110_CKA_c100_16/cifar100/scores7018.pt', map_location='cuda:%d' % args.gpu)
    else:
        assert "the model has not prepared"
    # if the model is loaded from torchvision, then the codes below do not need.
    if args.set in ['cifar10', 'cifar100']:
        if args.pretrained:
            model.load_state_dict(ckpt)
        else:
            print('No pretrained model')
    else:
        print('Not mentioned dataset')
    return model


def get_NAS_model(args, api):
    if args.arch[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        config = api.get_net_config(int(args.arch), args.set)
        model = get_cell_based_tiny_net(config)
        if args.pretrained:
            ckpt = torch.load('/public/ly/ASE/pretrained_model/tss_{}/cifar10/score.pt'.format(str(args.arch)), map_location='cuda:%d' % args.gpu)
            model.load_state_dict(ckpt)
    else:
        sys.exit(0)

    return model

