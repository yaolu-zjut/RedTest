# borrow from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202', 'resnet56_modularity_c10', 'resnet56_modularity_c100', 'resnet56_Shallowing', 'resnet56_Shallowing_c100', 'resnet56_CKA_c10_14', 'resnet110_CKA_c10_15', 'resnet1202_cka_15', 'resnet56_CKA_c100_15', 'resnet110_CKA_c100_16',
           'resnet402', 'resnet102']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes)

def resnet402(num_classes=10):
    return ResNet(BasicBlock, [150, 21, 29], num_classes=num_classes)

def resnet102(num_classes=10):
    return ResNet(BasicBlock, [26, 2, 22], num_classes=num_classes)

def resnet56_modularity_c10(num_classes=10):
    # [0.14904703925380478, 0.08492738294706721, 0.06794652311315814, 0.05842698842969518, 0.05774229988859838, 0.06661366419233368, 0.055583824362309284, 0.09609724991939289, 0.14926032887718313, 0.17190049735819515, 0.21193199833069842, 0.20496958339357002, 0.20395870053530177, 0.21940174019861672, 0.23310447632035985, 0.256061724329862, 0.288175910637973, 0.32357846644534155, 0.3503520543832207, 0.4274135258074049, 0.4413413978013006, 0.471266652511895, 0.5073605416426746, 0.5659299998297531, 0.6466699572680514, 0.7275012274478752, 0.7805404613816009, 0.809295637867689] quezhi 0.03
    return ResNet(BasicBlock, [2, 3, 6], num_classes=num_classes)

def resnet56_modularity_c100(num_classes=100):
    # according to [0.051385487023508064, 0.010230960668024746, 0.0016472184537852862, 0.027415129123244803, 0.02519807467603406, 0.03120689504113177, 0.03531556045536603, 0.043453019894257164, 0.04658137663832672, 0.04346871918898345, 0.07433679946116367, 0.07168839748348588, 0.07074251487696379, 0.07173137835899383, 0.07449850671622246, 0.07796990641968299, 0.07859764142526159, 0.09034319437772934, 0.09402030423169595, 0.12736248705785216, 0.14026570311477796, 0.19184588268453998, 0.2851119846986623, 0.45260194626259487, 0.5299878068105789, 0.5746359147645638, 0.5920711820247443, 0.5993459829348174]
    return ResNet(BasicBlock, [1, 6, 9], num_classes=num_classes) # to avoid first layer empty, we remain one

def resnet1202_cka_15(num_classes=10):
    return ResNet(BasicBlock, [8, 1, 6], num_classes=num_classes)

def resnet56_Shallowing(num_classes=10):
    # ref: Shallowing Deep Networks: Layer-wise Pruning based on Feature Representations
    # [33.73,51.11,52.67,54.36,58.55,58.4,58.98,62.16,63.2,64.38,70.5,72.78,73.91,75.5,76.75,78.1,79.2,80.77,82.02,83.77,85.88,87.41,89.09,90.38,91.84,92.73,93.06,93.07]
    return ResNet(BasicBlock, [6, 5, 5], num_classes=num_classes)

def resnet56_Shallowing_c100(num_classes=100):
    # ref: Shallowing Deep Networks: Layer-wise Pruning based on Feature Representations
    return ResNet(BasicBlock, [5, 5, 6], num_classes=num_classes)

def resnet56_CKA_c10_14(num_classes=10):  # remain 14 layers  used
    # according to
    # [0.8040034, 0.9623917, 0.9816055, 0.9818996, 0.9646843, 0.9950012, 0.96210957, 0.95488405, 0.9958612,
    #  0.9714489, 0.99030924, 0.9881511, 0.99465436, 0.99431086, 0.9923767, 0.9919683, 0.9893893, 0.9911849,
    #  0.9701058, 0.9823101, 0.9843625, 0.98163974, 0.966809, 0.93328685, 0.9032259, 0.8794835, 0.8777317]
    return ResNet(BasicBlock, [6, 1, 6], num_classes=num_classes)

def resnet56_CKA_c100_15(num_classes=100):  # remain 15 layers
    # according to
    # [0.89893186, 0.98167545, 0.8873335, 0.99292386, 0.99385583, 0.9889188, 0.992875, 0.9876197, 0.9502824,
    #  0.9415885, 0.97993964, 0.97814405, 0.98698264, 0.9881584, 0.9923223, 0.9799606, 0.98457384, 0.97167224,
    #  0.9489094, 0.9498153, 0.91059005, 0.8620137, 0.8641551, 0.9412409, 0.96756804, 0.9870943, 0.9951965]
    return ResNet(BasicBlock, [3, 4, 7], num_classes=num_classes)

def resnet110_CKA_c100_16(num_classes=100):  # remain 16 layers
    # according to
    # [0.93557775, 0.9792716, 0.95650053, 0.9941395, 0.9946949, 0.9853319, 0.99746835, 0.99607337, 0.99851835,
    #  0.9989565, 0.9962349, 0.9913951, 0.9932219, 0.9925826, 0.98335695, 0.99722975, 0.9600028, 0.99303514,
    #  0.9602537, 0.9886389, 0.99775743, 0.96976125, 0.991374, 0.9979828, 0.9915552, 0.9974679, 0.99442357,
    #  0.99303114, 0.99541825, 0.9949991, 0.99455535, 0.9932181, 0.995406, 0.9787676, 0.98938715, 0.9853932,
    #  0.9674503, 0.9751221, 0.9812791, 0.98032, 0.9749913, 0.9691599, 0.9447282, 0.9426031, 0.9150691,
    #  0.8937991, 0.9281022, 0.9595434, 0.97717637, 0.98760176, 0.99356556, 0.9965979, 0.9981247, 0.99893445]
    return ResNet(BasicBlock, [3, 2, 10], num_classes=num_classes)

def resnet110_CKA_c10_15(num_classes=10):  # remain 15 layers  used
    # according to
    # [0.94884205, 0.9770746, 0.95801353, 0.9939505, 0.98726845, 0.9893962, 0.98875153, 0.99533117, 0.997571,
    #  0.9886915, 0.97693574, 0.99091685, 0.9852234, 0.9873949, 0.9911769, 0.9832204, 0.9842418, 0.99356234,
    #  0.9645895, 0.99829596, 0.9973879, 0.99652565, 0.9983717, 0.9946502, 0.9971191, 0.9979045, 0.9978746,
    #  0.99804276, 0.9967415, 0.9954311, 0.99763125, 0.9974578, 0.996971, 0.99582374, 0.99504054, 0.9961294,
    #  0.982186, 0.9923633, 0.99419564, 0.99087715, 0.9947198, 0.99275845, 0.9914743, 0.99103945, 0.9932662,
    #  0.98934555, 0.98938334, 0.99056274, 0.98145306, 0.9834711, 0.965495, 0.9723147, 0.9525223, 0.89880216]
    return ResNet(BasicBlock, [6, 1, 7], num_classes=num_classes)

if __name__ == "__main__":
    model = resnet20()
    print(model)
    input =torch.rand((2, 3, 32, 32))
    print(model(input))
