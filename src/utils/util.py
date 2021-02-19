import cv2
import logging
import os
import torch
import yaml

import numpy as np
from colorlog import ColoredFormatter


# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


# MultiGPU
# ========

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Data
# ====

def normalization_params(data_name):
    if 'task1' in data_name:
        mean = [0., 0., 0.]
        std = [1., 1., 1.,]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return (mean, std)

def unnormalize_images(images, data_name):
    mean, std = normalization_params(data_name)
    mean = torch.reshape(torch.tensor(mean), (1, 3, 1, 1))
    std = torch.reshape(torch.tensor(std), (1, 3, 1, 1))
    unnormalized_images = images.clone().detach().cpu() * std + mean
    return unnormalized_images

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


# CAM
# ===
def cam(model, cam_topk=1, images=None, labels=None, truncate=False, shift=0.0):
    if images is not None:
        _ = model.forward(images)

    if labels is None:
        _, labels = torch.topk(
            model.pred, k=cam_topk, dim=1, largest=True, sorted=True)
        labels = labels[:, [cam_topk-1]]
    labels = labels.squeeze()

    last_layer = model.fc if model.last_layer == 'fc' else model.conv
    _, score_map = model.avgpool(
        model.feature_map, last_layer, truncate=truncate, shift=shift)
    cams = batched_index_select(score_map, 1, labels)
    return cams, labels


def extract_bbox(images, cams, gt_boxes, threshold=0.2, percentile=100,
                 color=[(0, 255, 0)]):
    # Convert the format of threshold and percentile
    if not isinstance(threshold, list):
        threshold = [threshold]
    if not isinstance(percentile, list):
        percentile = [percentile]

    assert len(threshold) == len(percentile)

    # Generate colors
    gt_color = (0, 0, 255) # (0, 0, 255)
    line_thickness = 2
    from itertools import cycle, islice
    color = list(islice(cycle(color), len(percentile)))

    # Convert a data format
    images = images.clone().numpy().transpose(0, 2, 3, 1)
    images = images[:, :, :, ::-1] * 255 # reverse the color representation(RGB -> BGR) and Opencv format
    cams = cams.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

    bboxes = []
    blended_bboxes = []
    for i in range(images.shape[0]):
        image, cam, gt_box = images[i].astype('uint8'), cams[i], gt_boxes[i]

        image_height, image_width, _ = image.shape
        cam = cv2.resize(cam, (image_height, image_width),
                         interpolation=cv2.INTER_CUBIC)

        # Generate a heatmap using jet colormap
        cam_max, cam_min = np.amax(cam), np.amin(cam)
        normalized_cam = (cam - cam_min) / (cam_max - cam_min) * 255
        normalized_cam = normalized_cam.astype('uint8')
        heatmap_jet = cv2.applyColorMap(normalized_cam, cv2.COLORMAP_JET)
        blend = cv2.addWeighted(heatmap_jet, 0.5, image, 0.5, 0)
        blended_bbox = blend.copy()
        if not isinstance(gt_box, str):
            cv2.rectangle(blended_bbox,
                          pt1=(gt_box[0], gt_box[1]), pt2=(gt_box[2], gt_box[3]),
                          color=gt_color, thickness=line_thickness)

        # Extract a bbox
        for _threshold, _percentile, _color in zip(threshold, percentile, color):
            threshold_val = int(_threshold * np.percentile(normalized_cam, q=_percentile))

            _, thresholded_gray_heatmap = cv2.threshold(
                normalized_cam, threshold_val, maxval=255, type=cv2.THRESH_BINARY)

            try:
                _, contours, _ = cv2.findContours(thresholded_gray_heatmap,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, _ = cv2.findContours(thresholded_gray_heatmap,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

            bbox = [0, 0, 224, 224]
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                bbox = [x, y, x + w, y + h]
                cv2.rectangle(blended_bbox,
                              pt1=(x, y), pt2=(x + w, y + h),
                              color=_color, thickness=line_thickness)

        blended_bbox = blended_bbox[:,:,::-1] / 255.0
        blended_bbox = blended_bbox.transpose(2, 0, 1)
        bboxes.append(torch.tensor(bbox))
        blended_bboxes.append(torch.tensor(blended_bbox))

    bboxes = torch.stack(bboxes)
    blended_bboxes = torch.stack(blended_bboxes)
    return bboxes, blended_bboxes

