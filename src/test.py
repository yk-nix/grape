import platform

from fastai.data.all import *
from fastai.vision.all import *


from cls_mnist import mnist_classifier
from lib.utils.misc import plot_metrics, load_metrics
from od_voc import voc_detector, get_dls_voc

def train_cls_mnist(start_epoch, end_epoch):
  learn = mnist_classifier(start_epoch=start_epoch)
  learn.fit(end_epoch, start_epoch=start_epoch)
  
def train_od_voc(start_epoch, end_epoch):
  learn = voc_detector(start_epoch=start_epoch)
  learn.fit(end_epoch, start_epoch=start_epoch) 
  
def show_plot_loss(model_name, start_epoch, end_epoch):
  learn = mnist_classifier()
  plot_metrics(learn, model_name, start_epoch=start_epoch, end_epoch=end_epoch)
  plt.show()
  
if __name__ == '__main__':
  sys = platform.system()
  wd = os.path.abspath('.')
  if sys == 'Linux' and  wd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {wd}')
  train_od_voc(0, 10)