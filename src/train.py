import platform
import os

from fastai.vision.all import *

from lib.models.ssd import SSD
from lib.exfastai.dataloaders import dls_voc
from lib.exfastai.learners import create_learner
from lib.exfastai.callbacks import AutoPlotCallback
from lib.exfastai.losses import SSDLossFlat


def dls_test(dls):
  train_dl, valid_dl = dls_voc()
  for i, x in enumerate(train_dl):
    print(i)
  for i, x in enumerate(valid_dl):
    print(i)

def voc_train():
  dls = dls_voc()  
  model = SSD(num_classes=len(dls.vocab), pretrained=False)
  loss_func = SSDLossFlat(ssd=model)
  opt_func = partial(SGD, mom=0.9)
  learn = create_learner('voc', dls, model, None, loss_func, opt_func)
  learn.fit(10, start_epoch=2)
  
if __name__ == '__main__':
  sys = platform.system()
  cwd = os.path.abspath('.')
  if sys == 'Linux' and  cwd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {cwd}')
  voc_train()