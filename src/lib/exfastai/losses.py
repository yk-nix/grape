import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert

from fastai.losses import *
from fastai.vision.all import *

from lib.models.ssd import SSDLoss
from lib.models.yolo_loss import YoloLoss
from .misc import post_predict

#-------------------------------------------------------------------------
class SSDLossFlat(BaseLoss):
  y_int = True # y interpolation
  def __init__(self, ssd, *args, axis=-1, nms_iou_threshold=0.3, score_threshold=0.95, top_n=200, **kwargs):
    self.nms_iou_threshold = nms_iou_threshold
    self.score_threshold = score_threshold
    self.top_n = top_n
    super().__init__(SSDLoss, ssd=ssd, *args, axis=axis, **kwargs)
    
  def __call__(self, preds, bboxes, labels):
    bboxes = bboxes.clamp(min=0, max=self.func.ssd.image_size)
    return self.func.__call__(preds, bboxes, labels)
    
  def activation(self, preds):
    conf, loc = preds
    scores = F.softmax(conf, dim=-1)
    bboxes_list = []
    for _bboxes in loc:
      bboxes_list.append(self.func.decode(_bboxes))
    bboxes = torch.stack(bboxes_list)
    return scores, bboxes
  
  def decodes(self, preds):
    _scores, _bboxes = preds
    #bboxes, labels, scores = post_predict(self, _scores[:,:,1:], _bboxes, self.top_n, self.score_threshold)
    bboxes, labels, scores = post_predict(_scores, _bboxes, self.top_n, self.nms_iou_threshold, self.score_threshold)
    return TensorBBox(bboxes), TensorMultiCategory(labels)
  
#-------------------------------------------------------------------------
class YoloLossFlat(BaseLoss):
  y_int = True   # y interpolation
  def __init__(self, model, *args, axis=-1, nms_iou_threshold=0.3, score_threshold=0.95, do_sqrt=True, **kwargs):
    self.nms_iou_threshold = nms_iou_threshold
    self.score_threshold = score_threshold
    self.num_classes = model.num_classes
    self.image_size = model.image_size
    self.size_w, self.size_h = model.grid_size
    super().__init__(YoloLoss, yolo=model, *args, axis=axis, do_sqrt=do_sqrt, **kwargs)
  
  def __call__(self, preds, bboxes, labels):
    labels = F.one_hot(labels, num_classes=self.num_classes)
    return self.func.__call__(preds, bboxes, labels)
  
  def decode(self, coord_preds, w, h):
    for idx in range(coord_preds.size(1)):
      i, j = idx / w, idx % w
      coord =  coord_preds[:,idx,:,:]
      coord[:,:,0] = (coord[:,:,0] + i)/w
      coord[:,:,1] = (coord[:,:,1] + j)/h
      if self.do_sqrt:
        coord[:,:,2:] = coord[:,:,2:].pow(2)
    return coord_preds.clamp(0.0, 1.0)
  
  def decodes(self, preds):
    b, (w, h), n, c = preds.size(0), self.func.yolo.grid_size, self.func.yolo.bbox_per_cell, self.func.yolo.num_classes
    cls_size, obj_size = w * h * c, w * h * n
    cls_scores, obj_scores, coord_preds = preds[:, :cls_size], preds[:, cls_size : cls_size + obj_size], preds[:, cls_size + obj_size :]
    cls_scores = cls_scores.view(b, w * h, 1, c)
    obj_scores = obj_scores.view(b, w * h, n, 1)
    coord_preds = coord_preds.view(b, w * h, n, -1)
    scores = (cls_scores * obj_scores).view(b, -1, c)
    bboxes = self.decode(coord_preds, w, h)
    bboxes = box_convert(bboxes, 'cxcywh', 'xyxy').clamp(min=0.0, max=1.0).view(b, -1, 4)
    bboxes, labels, scores = post_predict(scores, bboxes, 100, self.nms_iou_threshold, self.score_threshold, exclude_bg=False)
    return TensorBBox(bboxes), TensorMultiCategory(labels)
    
  def activation(self, preds):
    return preds
    