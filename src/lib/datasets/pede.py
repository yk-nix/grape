import os
import torch

from PIL import Image
from typing import Union, Optional, Callable, List, Tuple, Any, Dict
from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class PedDetection(VisionDataset):
  def __init__(self, 
               root: Union[str, Path],
               image_set: str = 'train',
               transform: Optional[Callable] = None,
               target_transform: Optional[Callable] = None,
               transforms: Optional[Callable] = None
  ):
    super().__init__(root, transforms, transform, target_transform)
    
    valid_image_sets = ['train', 'trainval', 'val']
    self.image_set = verify_str_arg(image_set, 'image_set', valid_image_sets)

    self.root = os.path.join(self.root, 'PennFudanPed')
    ped_root = os.path.join(self.root, 'PennFudanPed')

    if not os.path.isdir(ped_root):
      raise RuntimeError('Dataset not found or corrupted.')

    splits_dir = os.path.join(ped_root, 'ImageSets')
    split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
    with open(os.path.join(split_f)) as f:
      file_names = [x.strip() for x in f.readlines()]

    image_dir = os.path.join(ped_root, 'PNGImages')
    self.images = [os.path.join(image_dir, x + '.png') for x in file_names]

    target_dir = os.path.join(ped_root, 'Annotation')
    self.targets = [os.path.join(target_dir, x + '.txt') for x in file_names]

    assert len(self.images) == len(self.targets)

  def __len__(self) -> int:
    return len(self.images)
  

  # @property
  # def annotations(self) -> List[str]:
  #   return self.targets
  
  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is a dictionary of the .txt file
    """
    img = Image.open(self.images[index]).convert("RGB")
    target = self.parse_ped_txt(self.targets[index])

    if self.transforms is not None:
        img, target = self.transforms(img, target)

    return img, target
  
  def parse_ped_txt(self, txt: Union[str, Path]) -> Dict[str, Any]:
    ped_dict: Dict[str, Any] = {}
    ped_dict['labels'] = []
    ped_dict['boxes'] = []
    with open(txt) as f:
      for line in f.readlines():
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
          continue
        key, value = tuple(line.split(':')[:2])
        key = key.strip()
        value = value.strip().replace('"', '')
        if key == 'Image filename':
          ped_dict['filename'] = os.path.join(self.root, value)
        elif key == 'Image size (X x Y x C)':
          ped_dict['shape'] = [int(e.strip()) for e in value.split('x')]
        elif key.startswith('Bounding box for object'):
          ped_dict['boxes'].append([int(e.strip()) for e in value.translate(str.maketrans('()-','  ,')).split(',')])
          ped_dict['labels'].append(1)
      ped_dict['boxes'] = torch.tensor(ped_dict['boxes'])
      ped_dict['labels'] = torch.tensor(ped_dict['labels'])
    return ped_dict
