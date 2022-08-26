from fastai.data.all import *
from fastai.learner import *
from fastai.vision.all import *

from lib.utils import AutoSaveCallback, get_annotations_voc

__all__ = ['get_dls_voc','mnist_classifier']

def get_dls_voc(source: list = None):
  pass
  # if source is None:
  #   source = os.path.join(untar_data(URLs.MNIST), 'testing')
  # db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  # return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220822))


@delegates(Learner.__init__)
def voc_detector(data=None, model=None, lr=1e-3, model_name="od_voc", start_epoch=None, **kwargs):
  pass
  # if data is None: 
  #   data = get_dls_voc()
  # if model is None:
  #   model = LeNet()
  # learn = Learner(data, model, lr=lr, loss_func=CrossEntropyLossFlat(),
  #                opt_func=partial(SGD, mom=0.9), **kwargs)
  # if start_epoch and start_epoch > 0:
  #   weight_file_name = f'{model_name}/{start_epoch:03}'
  #   learn.load(weight_file_name)
  #   learn.start_epoch = start_epoch
  # else:
  #   learn.start_epoch = 0
  # learn.add_cbs([AutoSaveCallback(model_name='cls_mnist')])
  # return learn