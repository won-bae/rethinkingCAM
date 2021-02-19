import cv2
import torch
import numpy as np

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

# Object Localization
# ===================

def topk_accuracy(outputs, labels, topk=(1,)):
    """Computes the accuracy for the top k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = torch.topk(outputs, k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        topk_accuracies = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=False)
            topk_accuracies.append(correct_k.mul_(1.0 / batch_size).item())
        return topk_accuracies

def loc_accuracy(outputs, labels, gt_boxes, bboxes, iou_threshold):
    if outputs is not None:
        _, pred = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        wrongs = [c == 0 for c in correct.cpu().numpy()][0]

    batch_size = len(gt_boxes)
    gt_known, top1 = 0., 0.
    for i, (gt_box, bbox) in enumerate(zip(gt_boxes, bboxes)):
        iou_score = iou(gt_box, bbox)

        if iou_score >= iou_threshold:
            gt_known += 1.
            if outputs is not None and not wrongs[i]:
                top1 += 1.

    gt_loc = gt_known / batch_size
    top1_loc = top1 / batch_size
    return gt_loc, top1_loc

# Auxiliary for Object Localization
# =================================

def wrong_loc_index(outputs, labels, gt_boxes, bboxes, iou_threshold):
    if outputs is not None:
        _, pred = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        corrects = [c == 1 for c in correct.cpu().numpy()][0]

    indices = []
    for i, (gt_box, bbox) in enumerate(zip(gt_boxes, bboxes)):
        iou_score = iou(gt_box, bbox)
        if iou_score < iou_threshold or (not corrects[i]):
            indices.append(i)
    return torch.tensor(indices)

def area(boxes):
    xmin, ymin, xmax, ymax = boxes[:,0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)
    return areas.view(-1)

def iou(box1, box2):
    """box: (xmin, ymin, xmax, ymax)"""
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2

    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    inter_area = (inter_xmax - inter_xmin + 1) * (inter_ymax - inter_ymin + 1)
    box1_area = (box1_xmax - box1_xmin + 1) * (box1_ymax - box1_ymin + 1)
    box2_area = (box2_xmax - box2_xmin + 1) * (box2_ymax - box2_ymin + 1)

    iou = inter_area / (box1_area + box2_area - inter_area).float()
    return iou.item()

def multi_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious

def multi_gt_iou(boxes, box2):
    max_iou_score = 0.0
    for box in boxes:
        iou_score = iou(box, box2)
        if max_iou_score < iou_score:
            max_iou_score = iou_score
    return max_iou_score

def ioa(box1, box2):
    box1_xmin, box1_ymin, box1_xmax, box1_ymax = box1
    box2_xmin, box2_ymin, box2_xmax, box2_ymax = box2

    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    inter_area = (inter_xmax - inter_xmin + 1) * (inter_ymax - inter_ymin + 1)
    box1_area = (box1_xmax - box1_xmin + 1) * (box1_ymax - box1_ymin + 1)
    ioa = inter_area / box1_area.float()
    return ioa.item()

def ioa_between_masks(mask1, mask2):
    inter_masks = mask1.byte() & mask2.byte()
    inter_area = torch.sum(inter_masks)
    mask1_area = torch.sum(mask1)
    if torch.eq(mask1_area, 0.):
        ioa = mask1_area
    else:
        ioa = inter_area / mask1_area.float()
    return ioa.item()

def ioa_between_batch_masks(masks1, masks2):
    inter_masks = masks1.byte() & masks2.byte()
    inter_area = torch.sum(inter_masks, dim=(1,2)).float()
    masks1_area = torch.sum(masks1, dim=(1,2)).float()

    ioas = torch.where(torch.eq(masks1_area, 0.), masks1_area, inter_area / masks1_area)
    return ioas

