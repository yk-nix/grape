# -*- coding: utf-8 -*-
"""
Created on 2022.04.01

@author: yoka
"""
import torch
from math import sqrt
from torch import Tensor, arange, float32, meshgrid, cat, stack
from typing import List, Tuple


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def center_form(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    """bottom-right"""
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    """top-left"""
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    """(width, heigth)"""
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a: Tensor,    # shape: n x 4
            box_b: Tensor     # shape: m x 4
            ) -> Tensor :     # shape: n x m
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def encode(matched : Tensor,   # shape: n x 4
           priors : Tensor,    # shape: n x 4
           variances: List[float]
           ) -> Tensor :       # shape: n x 4
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def generate_cell_boxes(anchor_ratios: List,
                        anchor_size_limit: Tuple) -> Tensor:
    # i is the index of feature map
    boxes = []
    x, y = 0, 0
    # aspect_ratio: 1
    _min, _max = anchor_size_limit
    s = _min
    #boxes += [x-s/2, y-s/2, x+s/2, y+s/2]
    boxes += [x, y, x+s, y+s]
    s = sqrt(_min *_max)
    boxes += [x, y, x+s, y+s]
    #boxes += [x-s/2, y-s/2, x+s/2, y+s/2]
            
    # aspect_ratio: other value 
    for ratio in anchor_ratios:
        r = sqrt(ratio)
        a = r * _min
        b = (1/r) * _min
        #boxes += [x-a/2, y-b/2, x+a/2, y+b/2]
        #boxes += [x-b/2, y-a/2, x+b/2, y+a/2]
        boxes += [x, y, x+a, y+b]
        boxes += [x, y, x+b, y+a]
    return Tensor(boxes).view(-1, 4)

def generate_anchor_boxes(anchor_ratios: List[List],
                          feature_scales: List[Tuple],
                          feature_shapes: List[Tuple],
                          anchor_size_limits: List[Tuple],
                          image_size : int) -> Tensor:
    '''
        generate anchor boxes.
        shape of the return tensor is: n x 4
    '''
    if max(len(anchor_ratios), len(feature_scales), len(feature_shapes), len(anchor_size_limits)) != \
       min(len(anchor_ratios), len(feature_scales), len(feature_shapes), len(anchor_size_limits)):
        raise ValueError('anchor_ratios, feature_scales, feature_sizes, anchor_size_limits are not compitable.')
    boxes = []
    for i in range(len(feature_shapes)) :
        x, y = feature_shapes[i]
        scale_x, scale_y = feature_scales[i]
        shift_x = arange(0, x, dtype=float32) * scale_x
        shift_y = arange(0, y, dtype=float32) * scale_y
        shift_y, shift_x = meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        boxes.append((shifts.view(-1,1,4) + generate_cell_boxes(anchor_ratios[i], anchor_size_limits[i]).view(1,-1,4)).reshape(-1,4))
    boxes = cat(boxes).clamp_(0, image_size)
    return boxes

def nms(boxes: Tensor,     # shape: n x 4
        scores: Tensor,    # shape: n x num_classes  (0 is background)
        iou_threshold: float = 0.5,
        score_threshold: float = 0, 
        top_n: int = 200 ) -> Tensor :   # shape: n
    _, num_classes = scores.shape
    _, idx = scores.sort(dim=0)
    idx = idx[-top_n:]
    keep = []
    if boxes.numel() > 0:
        for c in range(1, num_classes):
            idxs = idx[:,c]
            while idxs.numel() > 0:
                i = idxs[-1]
                idxs = idxs[:-1]
                if score_threshold > scores[i, c]:
                    continue
                keep.append(i)
                if idxs.numel() == 1:
                    break
                iou = jaccard(torch.atleast_2d(boxes[i]), torch.atleast_2d(boxes[idxs])).flatten()
                idxs = idxs[iou <= iou_threshold]
    if len(keep) == 0:
        return torch.Tensor(), 0
    keep = torch.stack(keep).unique()
    return keep, keep.numel()
        
def match(truths: Tensor,  # shape: n x 4
          labels: Tensor,  # shape: n
          priors: Tensor,  # shape: m x 4
          iou_threshold: float = 0.5
          ) -> Tuple[Tensor] :
    overlaps = jaccard(truths, priors)
    best_prior_overlaps, best_prior_idxes = overlaps.max(dim=1)
    best_truth_overlaps, best_truth_idxes = overlaps.max(dim=0)
    for idx in best_prior_idxes:
        best_truth_overlaps[idx] += 1.0
    for i in range(len(best_prior_idxes)):
        idx = best_prior_idxes[i]
        if best_truth_overlaps[idx] < 2.0:
            best_truth_idxes[idx] = i        
    match_boxes, match_labels = truths[best_truth_idxes], labels[best_truth_idxes]
    match_labels[best_truth_overlaps < iou_threshold] = 0
    return match_labels, match_boxes
    
def log_sum_exp(x: Tensor      # n x m
                 ) -> Tensor:
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def hard_negative_mining(conf : Tensor,          # shape: n x num_classes
                         labels: Tensor,         # shape: n
                         negative_rate: float
                         ) -> Tensor :
    n, num_classes  = conf.shape
    positive = (labels > 0)
    num_positive = positive.sum()
    loss = log_sum_exp(conf) - conf.gather(1, labels.long().unsqueeze(1))
    loss[positive] = 0
    loss = loss.flatten()
    _, loss_idx = loss.sort(descending=True)
    _, idx_rank = loss_idx.sort()
    num_negative = torch.clamp(negative_rate * num_positive, max = n - 1)
    negative = idx_rank < num_negative
    return negative
    
    