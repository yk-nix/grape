
from typing import List, Tuple
import torch
from torch import Tensor, stack, cat, atleast_2d
from torch.nn import Module, Conv2d, Sequential, ReLU, BatchNorm2d, functional
from torchvision.ops.boxes import generalized_box_iou, box_convert
from math import sqrt

from .vgg import vgg16, vgg16_bn, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg19, vgg19_bn


__all__= ['SSD', 'SSDLoss']

_backbones = {
    'vgg11' : vgg11,
    'vgg13' : vgg13,
    'vgg16' : vgg16,
    'vgg19' : vgg19,
    'vgg11_bn' : vgg11_bn,
    'vgg13_bn' : vgg13_bn,
    'vgg16_bn' : vgg16_bn,
    'vgg19_bn' : vgg19_bn
}

def _make_layer(in_channels : int, 
                out_channels : int,
                stride: int,
                padding : int,
                batch_normal : bool = False) -> Sequential:
    conv1 = Conv2d(in_channels = in_channels, out_channels = int(out_channels/2), kernel_size = 1, stride = 1)
    bn1 = BatchNorm2d(int(out_channels/2))
    conv2 = Conv2d(in_channels = int(out_channels/2), out_channels = out_channels, kernel_size = 3, stride = stride, padding = padding)
    bn2 = BatchNorm2d((out_channels))
    relu = ReLU(inplace = True)
    if batch_normal :
      return Sequential(conv1, bn1, relu, conv2, bn2, relu)
    return Sequential(conv1, relu, conv2, relu)

def _make_layer0(in_channels : int,
                 out_channels : int,
                 batch_normal : bool = False) -> Sequential :
    conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
    conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1)
    relu = ReLU(inplace=True)
    bn  = BatchNorm2d(out_channels)
    if batch_normal:
      return Sequential(conv1, bn, relu, conv2, bn, relu)
    return Sequential(conv1, relu, conv2, relu)

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
        shift_x = torch.arange(0, x, dtype=torch.float32) * scale_x
        shift_y = torch.arange(0, y, dtype=torch.float32) * scale_y
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        boxes.append((shifts.view(-1,1,4) + generate_cell_boxes(anchor_ratios[i], anchor_size_limits[i]).view(1,-1,4)).reshape(-1,4))
    boxes = cat(boxes).clamp_(0, image_size)
    return boxes

class SSD(Module):
  '''
    SSD network, the following kwargs are supported:
      backbone: str: vgg16
      batch_normal: bool: False
      anchor_ratios: list : [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
      feature_scales: List[Tuple]: [(8,8), (16,16), (32,32), (64,64), (100,100), (304,304)],
      feature_shapes: List[Tuple]: [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)],
      anchor_size_limits: List[Tuple]: [(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)],
      image_size: int: 304
  '''
  def __init__(self, 
               num_classes:int,
               backbone:str='vgg16_bn',
               batch_normal:bool = True,
               image_size:int = 304,
               anchor_ratios:List = [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
               anchor_size_limits:List = [(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)],
               feature_shapes:List = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)],
               feature_scales:List = [(8,8), (16,16), (32,32), (64,64), (100,100), (304,304)],
               **kwargs):
    super(SSD, self).__init__()
    if num_classes <= 0:
        raise ValueError('invlaid num_classes: {}'.format(num_classes))
    num_dim = num_classes + 4
    self.num_classes = num_classes
    anchors = [len(e) * 2 + 2 for e in anchor_ratios]
    self.anchors = anchors
    self.anchor_ratios = anchor_ratios
    self.feature_shapes = feature_shapes
    self.num_classes = num_classes
    self.image_size = image_size
    self.feature_shapes = feature_shapes
    self.feature_scales = feature_scales
    self.anchor_size_limits = anchor_size_limits
    self.priors = generate_anchor_boxes(anchor_ratios, feature_scales, feature_shapes, anchor_size_limits, image_size)
    self.backbone = _backbones[backbone](**kwargs)
    self.name = f'ssd-{backbone}'
    self.pred0  = Conv2d(in_channels=512, out_channels = anchors[0] * num_dim, kernel_size=3, stride=1, padding=1)
    self.layer0 = _make_layer0(in_channels=512, out_channels=1024, batch_normal=batch_normal)
    self.pred1  = Conv2d(in_channels=1024, out_channels = anchors[1] * num_dim, kernel_size=3, stride=1, padding=1)
    self.layer1 = _make_layer(in_channels=1024, out_channels=512, stride=2, padding=1, batch_normal=batch_normal)
    self.pred2  = Conv2d(in_channels=512, out_channels = anchors[2] * num_dim, kernel_size=3, stride=1, padding=1)
    self.layer2 = _make_layer(in_channels=512, out_channels=256, stride=2, padding=1, batch_normal=batch_normal)
    self.pred3  = Conv2d(in_channels=256, out_channels = anchors[3] * num_dim, kernel_size=3, stride=1, padding=1)
    self.layer3 = _make_layer(in_channels=256, out_channels=256, stride=1, padding=0, batch_normal=batch_normal)
    self.pred4  = Conv2d(in_channels=256, out_channels = anchors[4] * num_dim, kernel_size=3, stride=1, padding=1)
    self.layer4 = _make_layer(in_channels=256, out_channels=128, stride=1, padding=0, batch_normal=batch_normal)
    self.pred5  = Conv2d(in_channels=128, out_channels = anchors[5] * num_dim, kernel_size=3, stride=1, padding=1)
    
  def refine_output(self, outputs: Tuple) -> Tuple[Tensor]:
    conf_list, loc_list = [], []
    for i in range(len(outputs)):
      o = outputs[i]
      b, c, w, h = o.shape
      n = self.anchors[i]
      m = self.num_classes + 4
      o = o.reshape(b, -1, h * w)
      o = o.permute(0, 2, 1)
      o = o.reshape(b, -1, m)
      conf_list.append(o[:,:,0:self.num_classes])
      loc_list.append(o[:,:,-4:])
    return cat(conf_list, dim=1), cat(loc_list, dim=1)
          
  def forward(self, x : Tensor) -> Tuple[Tensor]:
    '''
      x shape: b x c x 304 x 304  (only picture of size being 304x304 is supported)
      return: (conf,          shape: n x num_classes 
                loc)           shape: n x 4
    '''
    self.x = x
    x, x0 = self.backbone(x)
    out_0 = self.pred0(x0)
    x = self.layer0(x)
    out_1 = self.pred1(x)
    x = self.layer1(x)
    out_2 = self.pred2(x)
    x = self.layer2(x)
    out_3 = self.pred3(x)
    x = self.layer3(x)
    out_4 = self.pred4(x)
    x = self.layer4(x)
    out_5 = self.pred5(x)
    return self.refine_output((out_0, out_1, out_2, out_3, out_4, out_5))
  
  def load_pretrained_backbone(self) -> None:
    self.backbone = _backbones[self.backbone.name](pretrained=True)

class SSDLoss(Module):
  ''' SSD-loss function
  '''
  def __init__(self,
               ssd:SSD,
               negative_sample_rate:int = 0.5,      # ratio = negative / positive
               encode_variance:List = [1.0, 1.0],
               match_iou_threshold:float = 0.5) -> None:
    super(SSDLoss, self).__init__()
    if not isinstance(ssd, SSD):
      raise ValueError('SSDLoss is only supported for SSD, your model is {ssd.name}')
    self.ssd = ssd
    self.priors = box_convert(ssd.priors, 'xyxy', 'cxcywh')
    self.negative_sample_rate = negative_sample_rate
    self.variance = encode_variance
    self.match_iou_threshold = match_iou_threshold
    
  def match(self,
            _truths,   # shape: b x n x 4
            _labels,   # shape: b x n
            ):
    truths_list, labels_list = [], []
    for t, l in zip(_truths, _labels):
      iou = generalized_box_iou(t, self.ssd.priors)      # shape: n x m,  m=len(priors)
      _, best_box_idx = iou.max(dim=1)                   # shape: n
      best_label_score, best_label_idx = iou.max(dim=0)  # shape: m
      best_label_score[best_box_idx] += 1.0              # increment the score value for those boxes with best iou-values
      for i in range(len(best_box_idx)):
          best_label_idx[best_box_idx[i]] = i            # reassign the proper label-index to each boxes with best iou-values
      truths, labels = t[best_label_idx], l[best_label_idx]
      truths = self.encode(truths)
      labels[best_label_score < self.match_iou_threshold] = 0
      truths_list.append(truths)
      labels_list.append(labels)
    return cat(truths_list), cat(labels_list)
  
  def encode(self, matched : Tensor) -> Tensor :        # shape: n x 4
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
      matched: Shape: [num_priors, 4].  
    Return:
      encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - self.priors[:, :2]
    # encode variance
    g_cxcy /= (self.variance[0] * self.priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / self.priors[:, 2:]
    g_wh = torch.log(g_wh) / self.variance[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
    
  def decode(self, loc):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
      loc (tensor): location predictions for loc layers,
        Shape: [num_priors,4]
    Return:
      decoded bounding box predictions (xyxy-form)
    """
    boxes = torch.cat((
      self.priors[:, :2] + loc[:, :2] * self.variance[0] * self.priors[:, 2:],
      self.priors[:, 2:] * torch.exp(loc[:, 2:] * self.variance[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
  
  def hard_negative_mining(self,
                           conf : Tensor,          # shape: n x num_classes
                           labels: Tensor,         # shape: n
                          ) -> Tensor :
    n, num_classes  = conf.shape
    positive = (labels > 0)
    num_positive = positive.sum()
    loss = torch.logsumexp(conf, dim=1) - torch.gather(conf, dim=1, index=labels.long().unsqueeze(1)).flatten()
    loss[positive] = 0
    loss = loss.flatten()
    _, loss_idx = loss.sort(descending=True)
    _, idx_rank = loss_idx.sort()
    num_negative = torch.clamp(self.negative_sample_rate * num_positive, max = n - 1)
    negative = idx_rank < num_negative
    return negative
  
  def forward(self, _pred:Tuple, _bboxes:Tensor, _labels:Tensor) -> Tensor:
    conf, loc = _pred
    b, n, num_classes = conf.shape
    label_list, truth_list = [], []
    truths, labels = self.match(_bboxes, _labels)
    conf = conf.view(-1, num_classes)
    loc  = loc.view(-1, 4)
    negative = self.hard_negative_mining(conf, labels)
    positive = (labels > 0)
    loss_conf = functional.cross_entropy(conf[positive + negative], labels[positive + negative].long())
    loss_loc  = functional.smooth_l1_loss(loc[positive], truths[positive])
    return loss_conf + loss_loc
