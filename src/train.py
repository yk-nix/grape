import platform
import os

from fastai.vision.all import *

from lib.models.ssd import SSD, SSDLoss
from lib.exfastai.dataloaders import dls_voc
from lib.exfastai.learners import create_learner
from lib.exfastai.callbacks import AutoSaveCallback

def voc_train():
  dls = dls_voc()
  model = SSD(num_classes=len(dls.vocab))
  cbs = [AutoSaveCallback()]
  learn = create_learner('voc', dls, model, cbs, SSDLoss(model), partial(SGD, mom=0.9))
  learn.fit(10, start_epoch=1)
  
if __name__ == '__main__':
  sys = platform.system()
  cwd = os.path.abspath('.')
  if sys == 'Linux' and  cwd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {cwd}')
  voc_train()