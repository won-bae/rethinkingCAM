import torch
import torch.nn as nn
import numpy as np
from src.utils import util
from torch.utils.model_zoo import load_url
from src.core.models.avgpool import ThresholdedAvgPool2d, CustomAvgPool2d


MODEL_URLS = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

CONFIGS = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 1024],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 1024],
}


class VGG(nn.Module):
    def __init__(self, layers, **kwargs):
        super(VGG, self).__init__()

        self.num_classes = kwargs['num_classes']
        self.main_layers = nn.Sequential(*layers)
        self.bias = kwargs['bias']

        avgpool_threshold = kwargs['avgpool_threshold']
        if avgpool_threshold is not None:
            self.avgpool = ThresholdedAvgPool2d(avgpool_threshold)
        else:
            self.avgpool = CustomAvgPool2d()

        self.last_layer = kwargs['last_layer']
        if self.last_layer == 'fc':
            self.fc = nn.Linear(1024, self.num_classes, bias=self.bias)
        else:
            self.conv = nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0, bias=self.bias)

        if kwargs['init_weights']:
            self._initialize_weights()


    def forward(self, images):
        feature_map = self.main_layers(images)
        self.feature_map = feature_map

        if self.last_layer == 'fc':
            pred, _ = self.avgpool(feature_map, self.fc, bias=self.bias)
        else:
            pred, _ = self.avgpool(feature_map, self.conv, bias=self.bias)
        self.pred = pred

        results = {
            'preds': pred,
        }
        return results

    def downscale_gradient(self, ratio=0.1):
        last_bias, last_weight = True, True
        for name, param in reversed(list(self.main_layers.named_parameters())):
            if last_bias and 'bias' in name:
                last_bias = False
                continue
            if last_weight and 'weight' in name:
                last_weight = False
                continue
            param.grad *= ratio

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if isinstance(m, nn.Conv2d) and m.weight.shape[0] == self.num_classes:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(config, batch_norm=False, **params):
    layers = []

    in_channel = 3
    for i, c in enumerate(config):
        if c == 'M':
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            layers += [maxpool]
        else:
            conv2d = nn.Conv2d(in_channel, c, kernel_size=3, padding=1)

            if batch_norm:
                block =[conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                block = [conv2d, nn.ReLU(inplace=True)]
            layers += block
            in_channel = c
    return layers


def make_configs(config, **params):
    log = params['log']

    positions = []
    block, element = 1, 1
    for i in config:
        if isinstance(i, int):
            positions.append(str(block) + str(element))
            element += 1
        elif isinstance(i, str):
            positions.append(str(block) + 'M')
            block += 1
            element = 1

    new_config = []
    idx = 0
    for position in positions:
        new_config.append(config[idx])
        idx += 1

    log.infov('VGG network structure: {}'.format(new_config))
    return new_config


def remove_layers(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def make_config_align(state_dict, base_configs, target_configs, batch_norm=False):
    base_list, replace_list = [], []
    base_idx = 0
    for config in base_configs:
        if isinstance(config, int):
            base_list.append(base_idx)
            if batch_norm:
                base_idx += 3
            else:
                base_idx += 2
        elif isinstance(config, str):
            base_idx += 1
        else:
            raise Exception("No state_dict match")

    target_idx = 0
    for config in target_configs:
        if isinstance(config, int):
            replace_list.append(target_idx)
            if batch_norm:
                target_idx += 3
            else:
                target_idx += 2
        elif isinstance(config, str):
            target_idx += 1
        else:
            raise Exception("No state_dict match")

    target_list = {}
    for base, replace in zip(base_list, replace_list):
        target_list[str(base)] = str(replace)

    keys = [key for key in state_dict.keys()]
    for i, key in enumerate(reversed(keys)):
        key_ = key.split('.')
        new_key = key.replace('features', 'main_layers')
        if key_[1] in target_list:
            new_key = new_key.replace('.' + key_[1] + '.', '.' + target_list[key_[1]] + '.')
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def _vgg(arch, batch_norm, pretrained, progress, **kwargs):
    configs = make_configs(CONFIGS[arch], **kwargs)
    layers = make_layers(configs, batch_norm=batch_norm, **kwargs)

    model = VGG(layers, **kwargs)

    if pretrained:
        state_dict = load_url(MODEL_URLS[arch], progress=progress)
        state_dict = remove_layers(state_dict, 'classifier.')
        state_dict = make_config_align(state_dict, CONFIGS[arch], configs, batch_norm)
        strict = False
        model.load_state_dict(state_dict, strict=strict)
    return model


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = 'vgg16'
    batch_norm = kwargs.pop('batch_norm')
    if batch_norm:
        arch = 'vgg16_bn'
    return _vgg(arch=arch, batch_norm=batch_norm, pretrained=pretrained,
                progress=progress, **kwargs)

