from typing import Union, NoReturn, Callable, Any
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.ssd import ssd300_vgg16, SSDHead, SSD300_VGG16_Weights
from torchvision.models.detection._utils import retrieve_out_channels
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import StepLR

from .config import Configurable
from .learner import InteractiveLearner, print_logger, weight_saver
from lib.datasets.pede import PedDetection
from lib.uitls.vision import show_images

import torch
import torchvision.transforms.v2 as v2



class PedestrainDetector(Configurable):
  def __init__(self, config_file: Union[str, Path] = None):
    super().__init__(config_file)
    transforms = v2.Compose([v2.PILToTensor(),
                             v2.RandomHorizontalFlip(),
                             v2.RandomVerticalFlip(),
                             v2.ToDtype(torch.float, scale = True)])
    dataset = PedDetection(self.config.get('data_root'),
                           image_set='train',
                           transforms = transforms)
    dataloader = DataLoader(dataset, 
                            batch_size = int(self.config.get('batch_size', 4)),
                            shuffle = self.config.get('shuffle', True),
                            num_workers = int(self.config.get('num_workers', 0)),
                            collate_fn = lambda input: (list(e) for e in zip(*input)))     
    ssd = ssd300_vgg16(weights = SSD300_VGG16_Weights.DEFAULT,
                       trainable_backbone_layers = 0,
                       score_thresh = float(self.config.get('score_thresh', 0.9)))
    out_channels = retrieve_out_channels(ssd.backbone, size = (300, 300))
    num_anchors = ssd.anchor_generator.num_anchors_per_location()
    ssd.head = SSDHead(out_channels, num_anchors, num_classes = 2)
    optimizer = SGD(ssd.head.parameters(),
                    lr = float(self.config.get('learning_rate', 0.01)),
                    momentum = float(self.config.get('momentum', 0.9)))
    lr_scheduler = StepLR(optimizer,
                          step_size = int(self.config.get('step_size', 10)),
                          gamma = float(self.config.get('gama', 0.8)))
    self.learner = InteractiveLearner(port = int(self.config.get('port', 12345)),
                                      root = self.config.get('weight_root'),
                                      name = 'ssd300_vgg16_pede',
                                      train_dataloader = dataloader,
                                      eval_dataloader = None,
                                      model = ssd,
                                      loss_fn = lambda output, target: output['bbox_regression'] + output['classification'],
                                      eval_fn = lambda x: x,
                                      optimizer = optimizer,
                                      lr_scheduler= lr_scheduler,
                                      loop_cb = lambda learner: print_logger(learner = learner, freq = int(self.config.get('logger_freq', 1))),
                                      epoch_cb = lambda learner: weight_saver(learner = learner, freq = int(self.config.get('saver_freq', 20))))
  
  def train(self, 
            weight_file: Union[str, Path]= None,
            load_cb: Callable = None,
            **kwargs: Any ) -> NoReturn:
    if weight_file is not None:
      self.learner.load(weight_file, **kwargs)
      if load_cb is not None:
        load_cb(self)
    self.learner.train()

  def test(self, weight_file: Union[str, Path])-> NoReturn:
    transform = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float, scale = True)])
    dataset = PedDetection(self.config.get('data_root'),
                           image_set='val',
                           transform = transform)
    self.learner.load(weight_file,
                      ignore_optimizer = True,
                      ignore_lr_scheduler = True)
    for x, _ in dataset:
      p = self.learner.predict([x])[0]
      print(p['scores'])
      boxes = p['boxes'].to(torch.int)
      show_images([x], [boxes], width = 2)