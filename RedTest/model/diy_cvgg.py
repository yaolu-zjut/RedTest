import math
import torch.nn as nn


__all__ = ['VGG', 'VGG19_CKA_c10', 'cvgg19_modularity_c10', 'VGG19_Shallowing_c10', 'VGG19_CKA_c10', 'VGG16_CKA_c10']


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),  # 112 or 512
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'CKA_19_c10': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M'],
    'CKA_19_c100': [64, 'M', 128, 128, 'M', 256, 'M', 512, 512,'M', 512, 512, 'M'],
    'CKA_16_c10': [64, 'M', 128, 128, 'M', 256, 'M', 512, 512, 'M', 512, 'M'],
    'modularity_19_c10': [64, 'M', 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 'M'],
    'Shallowing_19_c10': [64, 64, 'M', 128, 'M', 256, 256, 256, 256, 'M', 512, 'M', 512, 'M'],
    'Shallowing_16_c100': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'M', 512, 'M'],
}

def VGG19_Shallowing_c10(num_classes=10):
    # 35.06,42.89,64.25,64.65,73.06,75.62,79.97,85.81,91.17,92.83,93.16,93.41,93.3,93.34,93.41,93.41
    return VGG(make_layers(cfg['Shallowing_19_c10'], batch_norm=True), num_classes=num_classes)

def VGG19_CKA_c10(num_classes=10):
    # [0.81111956, 0.5691167, 0.7536302, 0.7598313, 0.84407616, 0.8530037, 0.7927989, 0.79135907,
    #  0.77015305, 0.88825816, 0.77569777, 0.7998207, 0.91992533, 0.86013335, 0.9487301]
    return VGG(make_layers(cfg['CKA_19_c10'], batch_norm=True), num_classes=num_classes)

def VGG16_CKA_c10(num_classes=10):
    # [0.8088623, 0.558723, 0.75259703, 0.7198383, 0.78371954, 0.8439565,
    #  0.74877006, 0.74864566, 0.8681474, 0.75026524, 0.79262555, 0.8751615]
    return VGG(make_layers(cfg['CKA_16_c10'], batch_norm=True), num_classes=num_classes)

def cvgg19_modularity_c10(num_classes):
    # [0.14684793347214528, 0.04648090105265012, 0.23842332137255468, 0.16419156223094114, 0.41783892321376787, 0.45425065945253196, 0.5439824592699893, 0.6443418382530606, 0.7832348765139164, 0.8060550687498027, 0.8127366200019036, 0.8172894481281731, 0.8065141932310893, 0.8027318229062533, 0.7996691706581778, 0.7965051868929202]
    return VGG(make_layers(cfg['modularity_19_c10'], batch_norm=True), num_classes=num_classes)

if __name__ == "__main__":
    import torch
    input = torch.randn((2, 3, 32, 32))
    model = cvgg19_modularity_c10(100)
    print(model)
    output = model(input)
    print(output)



