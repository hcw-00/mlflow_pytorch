# import collections
# import argparse
import torch
from torch import nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from torch.autograd import Function
# import torchvision
# from torchvision import models, transforms, datasets
# from torch.utils.data import Dataset, DataLoader
# import os
# import warnings
# import shutil
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import time
# from torch.utils.tensorboard import SummaryWriter
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'HDL': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'], # vgg16 modified(for HDL) : conv1 ~ 4, remove conv5
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.features = features # size : (14, 14) #?
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # original : (7,7)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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


def vgg16(pretrained, arch='vgg16', cfg='HDL', batch_norm=False, progress=True, num_classes=4, **kwargs):    
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes, **kwargs)
    model_dict = model.state_dict()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # filter out unnecessary keys & update weights
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "classifier" not in k}
        model_dict.update(state_dict) 
        model.load_state_dict(model_dict)

    return model