import platform

from fastai.data.all import *
from fastai.vision.all import *


from cls_mnist import mnist_classifier
from lib.utils import plot_metrics, load_metrics

def main():
  start_epoch = 16
  learn = mnist_classifier(start_epoch=start_epoch)
  learn.fit(30, start_epoch=start_epoch)
  pass
  
def show_plot_loss():
  learn = mnist_classifier()
  plot_metrics(learn, 'cls_mnist', start_epoch=5, end_epoch=30)
  plt.show()
  
if __name__ == '__main__':
  sys = platform.system()
  wd = os.path.abspath('.')
  if sys == 'Linux' and  wd != '/data/grapefruits':
    raise ValueError(f'Wrong working directory: {wd}')
  main()