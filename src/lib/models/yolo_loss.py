from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce
from torchvision.ops.boxes import box_convert, generalized_box_iou

from .misc import BasicConv2d, BasicLinear, BasicModule

class YoloLoss(nn.Module):
  def __init__(self, yolo, noobject_scale: float=0.5, object_scale: float=1.0, class_scale: float=1.0, coord_scale: float=5.0, do_sqrt=True):
    super().__init__()
    self.yolo = yolo
    self.noobject_scale = noobject_scale
    self.object_scale = object_scale
    self.class_scale = class_scale
    self.coord_scale = coord_scale
    self.loss_func = nn.MSELoss()
    self.do_sqrt = do_sqrt
  
  def get_truths(self, bboxes, labels):
    cls_score_list, obj_score_list, coord_truth_list = [], [], []
    (w, h), num_classes, image_size = self.yolo.grid_size, self.yolo.num_classes, self.yolo.image_size
    for _bboxes, _labels in zip(bboxes, labels):
      cls_scores = torch.zeros(h, w, num_classes, dtype=torch.float)
      obj_scores = torch.zeros(h, w, dtype=torch.float)
      coord_truths = torch.zeros(h, w, 4, dtype=torch.float)
      _bboxes = box_convert(_bboxes, 'xyxy', 'cxcywh') / image_size
      x = (_bboxes[:,0].mul_(w)).long()
      y = (_bboxes[:,1].mul_(h)).long()
      _bboxes[:,0] -= x
      _bboxes[:,1] -= y
      if self.do_sqrt:
        _bboxes[:,2:].sqrt_()
      mask = torch.zeros(h, w, dtype=torch.bool)
      mask.flatten()[y * w + y] = 1
      count = mask.sum()
      cls_scores[mask] = _labels[:count].float()
      obj_scores[mask] = 1
      coord_truths[mask] = _bboxes[:count]
      cls_score_list.append(cls_scores)
      obj_score_list.append(obj_scores)
      coord_truth_list.append(coord_truths)
    return torch.stack(cls_score_list), torch.stack(obj_score_list), torch.stack(coord_truth_list)
  
  def select_bbox(self, coord_preds, coord_truths, mask, n):
    mask_list = []
    for b, x, y in zip(*torch.where(mask > 0)):
      m = torch.zeros(*mask.size(), n, dtype=torch.bool)
      pred = coord_preds[b, x, y].clone()
      truth = coord_truths[b, x, y][None].clone()
      if self.do_sqrt:
        pred[:,2:].pow_(2)
        truth[:,2:].pow_(2)
      pred = box_convert(pred, 'cxcywh', 'xyxy')
      truth = box_convert(truth, 'cxcywh', 'xyxy')
      i = generalized_box_iou(pred, truth).argmax()
      m[b, x, y, i] = 1
      mask_list.append(m)
    return reduce(lambda x, y: x + y, mask_list)
        
  def forward(self, preds, bboxes, labels):
    cls_truths, obj_truths, coord_truths = self.get_truths(bboxes, labels)
    b, (w, h), n, c = preds.size(0), self.yolo.grid_size, self.yolo.bbox_per_cell, self.yolo.num_classes
    cls_size, obj_size = h * w * c, h * w * n
    cls_scores, obj_scores, coord_preds = preds[:, :cls_size], preds[:, cls_size: cls_size + obj_size], preds[:, cls_size + obj_size:]
    cls_scores = cls_scores.view(b, h, w, c)
    obj_scores = obj_scores.view(b, h, w, n)
    coord_preds = coord_preds.view(b, h, w, n, 4)
    mask = obj_truths.bool()
    cls_loss = self.loss_func(cls_truths[mask], cls_scores[mask]) * self.class_scale
    masks = self.select_bbox(coord_preds, coord_truths, mask, n)
    obj_truths = masks.float()
    obj_loss = self.loss_func(obj_truths[masks], obj_scores[masks]) * self.object_scale
    noobj_loss = self.loss_func(obj_truths[~masks], obj_scores[~masks]) * self.noobject_scale
    coord_loss = self.loss_func(coord_truths[mask], coord_preds[masks]) * self.coord_scale
    return cls_loss  + obj_loss + noobj_loss + coord_loss