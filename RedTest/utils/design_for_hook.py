from model.VGG_cifar import *
from args import args

cfgs = {
    'ResNet18': [2, 2, 2, 2],
    'ResNet34': [3, 4, 6, 3],
    'ResNet50': [3, 4, 6, 3],
    'ResNet101': [3, 4, 23, 3],
    'ResNet152': [3, 8, 36, 3],
    'resnet20': [3, 3, 3],
    'resnet32': [5, 5, 5],
    'resnet56_CKA_c10_14': [6, 1, 6],
    'resnet44': [7, 7, 7],
    'resnet56': [9, 9, 9],
    'resnet110': [18, 18, 18],
    'resnet1202': [200, 200, 200],
    'resnet402': [150, 21, 29],
    'resnet102': [26, 2, 22],
    'resnet1202_cka_15': [8, 1, 6],
    'Ivgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22', 'features.25'],
    'Ivgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21', 'features.24', 'features.28', 'features.31'],
    'Ivgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20', 'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'Ivgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17','features.20', 'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43', 'features.46', 'features.49'],
    'cvgg11_bn': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15', 'features.18', 'features.22',
               'features.25'],
    'cvgg13_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.21',
               'features.24', 'features.28', 'features.31'],
    'cvgg16_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.24', 'features.27', 'features.30', 'features.34', 'features.37', 'features.40'],
    'cvgg19_bn': ['features.0', 'features.3', 'features.7', 'features.10', 'features.14', 'features.17', 'features.20',
               'features.23', 'features.27', 'features.30', 'features.33', 'features.36', 'features.40', 'features.43',
               'features.46', 'features.49'],
    'cvgg16_5': ['features.0', 'features.4', 'features.8', 'features.11', 'features.14', 'features.18', 'features.21',
                 'features.25'],
    'resnet56_modularity_c10': [2, 3, 6],
    'resnet56_Shallowing': [6, 5, 5],
    'VGG19_CKA_c10': ['features.0', 'features.4', 'features.7', 'features.11', 'features.14', 'features.18', 'features.21', 'features.25'],
    'VGG19_modularity_c10': ['features.0', 'features.4', 'features.8', 'features.11', 'features.14', 'features.17', 'features.21', 'features.24', 'features.27', 'features.31'],
    'VGG19_Shallowing_c10': ['features.0', 'features.3', 'features.7', 'features.11', 'features.17', 'features.20', 'features.24', 'features.28'],
    'NATS-Bench_tss': ['stem.0'] + ['cells.%d' % i for i in range(0, 17)],
}

def get_inner_feature_for_resnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)  # here!!!
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[3]):
        handle = model.layer4[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_smallresnet(model, hook, arch):
    handle_list = []
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]
    print('cfg:', cfg)
    handle = model.conv1.register_forward_hook(hook)
    handle_list.append(handle)
    # handle.remove()  # free memory
    for i in range(cfg[0]):
        handle = model.layer1[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[1]):
        handle = model.layer2[i].register_forward_hook(hook)
        handle_list.append(handle)
    for i in range(cfg[2]):
        handle = model.layer3[i].register_forward_hook(hook)
        handle_list.append(handle)
    return handle_list


def get_inner_feature_for_vgg(model, hook, arch):
    cfg = cfgs[arch]
    if args.multigpu is not None:
        for i in range(len(cfg)):
            cfg[i] = 'module.' + cfg[i]

    handle_list = []
    print('cfg:', cfg)
    count = 0
    for idx, m in enumerate(model.named_modules()):
        name, module = m[0], m[1]
        if count < len(cfg):
            if name == cfg[count]:
                print(module)
                handle = module.register_forward_hook(hook)
                handle_list.append(handle)
                count += 1
        else:
            break
    return handle_list


if __name__ == "__main__":
    # demo
    import torch
    from torchvision.models import *

    input = torch.randn((2, 3, 224, 224))
    inter_feature = []
    model = vgg11_bn()

    def hook(module, input, output):
        inter_feature.append(output.clone().detach())
    get_inner_feature_for_vgg(model, hook, 'cvgg19')
    model(input)
