import torch
import torch.nn as nn

class CustomAvgPool2d(nn.Module):
    def __init__(self):
        super(CustomAvgPool2d, self).__init__()

    def forward(self, feature_map, layer, truncate=False, shift=0.0, bias=True):
        _, _, height, width = feature_map.shape

        avg_feature_map = feature_map / (height * width)

        weight = layer.weight + compute_shift(shift, layer.weight)

        if truncate:
            weight = torch.where(torch.gt(layer.weight, 0.),
                                 layer.weight, torch.zeros_like(layer.weight))

        if len(weight.shape) < 4:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        score_map = nn.functional.conv2d(
            avg_feature_map, weight=weight, bias=None)
        pred = torch.sum(score_map, dim=(2, 3))

        if bias and layer.bias is not None:
            pred = pred + layer.bias.view(1, -1)

        return pred, score_map


class ThresholdedAvgPool2d(nn.Module):
    def __init__(self, threshold=0.0):
        super(ThresholdedAvgPool2d, self).__init__()
        self.threshold = threshold

    def forward(self, feature_map, layer, truncate=False, shift=0.0, bias=True):
        # threshold feature map
        batch_size, channel, height, width = feature_map.shape
        max_vals, _ = torch.max(feature_map.view(batch_size, channel, -1), dim=2)
        thr_vals = (max_vals * self.threshold).view(batch_size, channel, 1, 1).expand_as(feature_map)
        thr_feature_map = torch.where(
            torch.gt(feature_map, thr_vals), feature_map, torch.zeros_like(feature_map))

        # divided by the number of positives
        num_positives = torch.sum(torch.gt(thr_feature_map, 0.), dim=(2,3))
        num_positives = torch.where(torch.eq(num_positives, 0),
                                    torch.ones_like(num_positives),
                                    num_positives).view(batch_size, channel, 1, 1).expand_as(feature_map)
        avg_feature_map = torch.div(thr_feature_map, num_positives.float())

        # convolve
        #weight = layer.weight + compute_shift(shift, layer.weight)
        weight = layer.weight

        if truncate:
            weight = torch.where(torch.gt(layer.weight, 0.),
                                 layer.weight, torch.zeros_like(layer.weight))

        if len(weight.shape) < 4:
            weight = weight.unsqueeze(-1).unsqueeze(-1)

        avgpooled_map = nn.functional.conv2d(
            avg_feature_map, weight=weight, bias=None)
        pred = torch.sum(avgpooled_map, dim=(2,3))
        score_map = nn.functional.conv2d(
            feature_map, weight=weight, bias=None)
        if bias:
            pred = pred + layer.bias.view(1, -1)

        return pred, score_map

def compute_shift(shift, weight):
    shift_type = list(shift.keys())[0]
    if shift_type == 'global_min':
        shift = torch.abs(torch.min(weight)) + float(shift[shift_type])
    elif shift_type == 'adaptive_min':
        shift = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1)) + float(shift[shift_type])
    elif shift_type == 'global_multiple':
        gmin = torch.min(weight)
        gmax = torch.max(weight)
        k = shift[shift_type]
        shift = (gmax - k * gmin) / float(k - 1)
    elif shift_type == 'adaptive_multiple':
        amin = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1))
        amax = torch.abs(torch.min(weight, dim=1)[0].unsqueeze(-1))
        k = shift[shift_type]
        shift = (amax - k * amin) / float(k - 1)

    return shift
