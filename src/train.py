import platform
import os

from fastai.vision.all import *

from lib.models.all import *
from lib.exfastai.all import *


def dls_test(dls):
  train_dl, valid_dl = dls_voc()
  for i, x in enumerate(train_dl):
    print(i)
  for i, x in enumerate(valid_dl):
    print(i)

def voc_train():
  dls = dls_voc(list_dir=os.path.join('ImageSets', 'Layout'))  
  model = SSD(num_classes=len(dls.vocab), pretrained=False)
  loss_func = SSDLossFlat(ssd=model)
  opt_func = partial(SGD, mom=0.9)
  learn = create_learner('voc', dls, model, None, loss_func, opt_func)
  learn.fit(20, start_epoch=0)
  
if __name__ == '__main__':
  sys = platform.system()
  cwd = os.path.abspath('.')
  if sys == 'Linux' and  cwd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {cwd}')
  voc_train()