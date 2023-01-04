from Data.ImageNet_dali import ImageNetDali
from Data.load_data import *
from Data.ImageNet16_120 import ImageNet16_120


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    if args.set == 'imagenet_dali':
        dataset = ImageNetDali()
    elif args.set == 'cifar10':
        dataset = CIFAR10()  # for normal training
    elif args.set == 'cifar100':
        dataset = CIFAR100()  # for normal training
    elif args.set == 'ImageNet16-120':
        dataset = ImageNet16_120()  # for normal training
    return dataset