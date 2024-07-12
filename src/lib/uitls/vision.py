from typing import Tuple, Sequence, List, Dict, NoReturn, Union, Optional
from math import ceil, sqrt
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as tf

__all__ = ['show_images']

def caculate_grid_size(n: int) -> Tuple[int]:
  w = ceil(sqrt(n * 1.5))
  h = ceil(n / w)
  if h == 1:
    w = n
  return (h, w)

def unnormalize(images: Tensor, 
                image_mean: List[float],
                image_std: List[float],) -> Tensor:
  image_mean = torch.tensor(image_mean).view(1, -1, 1, 1)
  image_std = torch.tensor(image_std).view(1, -1, 1, 1)
  return images * image_std + image_mean

def show_images(images: Sequence[Tensor], 
                boxes: Sequence[Tensor] = None, 
                labels: Sequence[List[str]] = None,
                colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
                width: int = 1,
                font: Optional[str] = None,
                font_size: Optional[int] = None) -> NoReturn:
  N = len(images)
  rows, colums = caculate_grid_size(N)
  for i in range(0, N, 1):
    boundBoxes = None if boxes is None or len(boxes) == 0 else boxes[i]
    labelNames = None if labels is None or len(labels) == 0 else labels[i]
    image = images[i]
    # colors = ['red'] * len(boundBoxes)
    if image.dtype != torch.uint8:
      image = tf.to_dtype(image, torch.uint8, scale = True)
    if boundBoxes is not None:
      image = draw_bounding_boxes(image, (boundBoxes - width).clamp_min(0), labelNames,
                                  colors = colors, width = width, font = font, font_size = font_size)
    ax = plt.subplot(rows, colums, i + 1, frameon = False)
    ax.set_axis_off()
    ax.imshow(tf.to_pil_image(image))
  plt.show()