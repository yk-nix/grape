from fastai.vision.all import *

from lib.models.lenet import LeNet
from lib.models.ssd import SSD, SSDLoss
from lib.exfastai.misc import point_scale_boxes
from lib.exfastai.callbacks import AutoPlotCallback, AutoSaveCallback
from lib.exfastai.dataloaders import dls_mnist, dls_voc
from lib.exfastai.learners import create_learner



def mnist_train_test():
  dls = dls_mnist(num_workers=0)
  model = LeNet()
  cbs = [AutoSaveCallback()]
  learn = create_learner('mnist', dls, model, cbs, loss_func=CrossEntropyLossFlat())
  learn.plot(30)
  plt.show()
  #learn.fit(30, start_epoch=20)

def voc_train_test():
  dls = dls_voc(num_workers=0)
  model = SSD(num_classes=21)
  model.priors = point_scale_boxes(model.priors, model.image_size)
  cbs = [AutoSaveCallback()]
  learn = create_learner('voc', dls, model, cbs, loss_func=SSDLoss(model))
  learn.fit(1)

if __name__ == '__main__':
  mnist_train_test()
