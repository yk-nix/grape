from torch import Tensor
from typing import List, Optional, Callable, Dict, Any, Tuple
from PIL.Image import Image
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tf

__all__ = ['voc_classes_num', 'voc_name2label', 'voc_label2name', 'voc_detection_transforms_wrapper', 'voc_detection_collate_fn']

_unkown = {'name': 'unkown',  'label': 0 }

_voc_classes = [
  {'name': 'aeroplane',   'label': 1 },
  {'name': 'bicycle',     'label': 2 },
  {'name': 'bird',        'label': 3 },
  {'name': 'boat',        'label': 5 },
  {'name': 'bottle',      'label': 5 },
  {'name': 'bus',         'label': 6 },
  {'name': 'car',         'label': 7 },
  {'name': 'cat',         'label': 8 },
  {'name': 'chair',       'label': 9 },
  {'name': 'cow',         'label': 10 },
  {'name': 'diningtable', 'label': 11 },
  {'name': 'dog',         'label': 12 },
  {'name': 'horse',       'label': 13 },
  {'name': 'moterbike',   'label': 14 },
  {'name': 'person',      'label': 15 },
  {'name': 'pottedplant', 'label': 16 },
  {'name': 'sheep',       'label': 17 },
  {'name': 'sofa',        'label': 18 },
  {'name': 'train',       'label': 19 },
  {'name': 'tvmonitor',   'label': 20 }
]

voc_classes_num = len(_voc_classes) + 1

def voc_name2label(name: str) -> Tensor:
  for elem in _voc_classes:
    if (elem['name'] == name):
      return torch.tensor([elem['label']])
  return torch.tensor([0])

def voc_label2name(labels: Tensor) -> List[str]:
  names = []
  for label in labels.tolist():
    for elem in _voc_classes:
      if (label == elem['label']):
        names += [elem['name']]
        break
    names += [_unkown['name']]
  return names


__voc_dectection_tranforms = None

def _voc_detection_transforms(image: Image, annotation: Dict) -> Any:
  boundboxes = [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), 
                 int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])]
                for obj in annotation['annotation']['object']]
  labels = torch.tensor([voc_name2label(obj['name']) for obj in annotation['annotation']['object']])
  H = int(annotation['annotation']['size']['height'])
  W = int(annotation['annotation']['size']['width'])
  boxes = BoundingBoxes(boundboxes, format = BoundingBoxFormat.XYXY, canvas_size = (H, W))
  if __voc_dectection_tranforms:
    image, boxes = __voc_dectection_tranforms((tf.pil_to_tensor(image), boxes))
  annotation['boxes'] = boxes
  annotation['labels'] = labels
  return image, annotation

def voc_detection_transforms_wrapper(transforms: Optional[Callable] = None) -> Callable:
  global __voc_dectection_tranforms
  __voc_dectection_tranforms = transforms
  return _voc_detection_transforms


def voc_detection_collate_fn(samples: Tuple[Tensor, Dict]) -> Tuple[Tensor, Tensor]:
  images, annotations = zip(*samples)
  return torch.stack(images), list(annotations)