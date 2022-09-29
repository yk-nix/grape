from fastai.vision.all import *

from lib.models.lenet import LeNet
from lib.models.ssd import SSD

from lib.exfastai.misc import point_scale_boxes
from lib.exfastai.callbacks import AutoPlotCallback, AutoSaveCallback
from lib.exfastai.dataloaders import dls_mnist, dls_voc
from lib.exfastai.learners import create_learner
from lib.exfastai.losses import SSDLossFLat



def mnist_train_test():
  dls = dls_mnist(num_workers=0)
  model = LeNet(num_class=len(dls.vocab))
  cbs = [AutoSaveCallback()]
  loss_func = CrossEntropyLossFlat()
  opt_func = partial(SGD, mom=0.9)
  learn = create_learner('mnist', dls, model, cbs, loss_func, opt_func)
  
  learn.load('8')
  learn.show_results()
  plt.show()
  
  # source = get_image_files(Path(r'C:\Users\yui\.fastai\data\mnist_png\testing'))
  # res = learn.predict(source[1000])
  # print(res)
  
  # learn.plot(9, with_lr=False)
  # plt.show()
  #learn.fit(10, start_epoch=8)



def voc_train_test():
  dls = dls_voc(Path(r'D:\grapefruits\data\VOC\VOCdevkit\VOC2007'))
  model = SSD(num_classes=len(dls.vocab))
  model.priors = point_scale_boxes(model.priors, model.image_size)
  cbs = [AutoSaveCallback()]
  loss_func = SSDLossFLat(ssd=model)
  opt_func = partial(SGD, mom=0.9)
  learn = create_learner('voc', dls, model, cbs, loss_func, opt_func)
  
  learn.load('2')
  learn.show_results()
  plt.show()
  
  # learn.plot(3)
  # plt.show()
  #learn.fit(1, start_epoch=0)



  

if __name__ == '__main__':
  pass