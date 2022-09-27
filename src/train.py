import platform
import os

from fastai.vision.all import *

from lib.models.ssd import SSD, SSDLoss
from lib.exfastai.dataloaders import dls_voc
from lib.exfastai.learners import create_learner
from lib.exfastai.callbacks import AutoSaveCallback

def voc_train():
  # source: Path('D:\grapefruit\data\VOC\VOCdevkit\VOC2007')
  dls = dls_voc()
  model = SSD(num_classes=21)
  cbs = [AutoSaveCallback()]
  learn = create_learner('voc', dls, model, cbs, SSDLoss(model))
  learn.fit(1)
  
if __name__ == '__main__':
  sys = platform.system()
  wd = os.path.abspath('.')
  if sys == 'Linux' and  wd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {wd}')
  voc_train()