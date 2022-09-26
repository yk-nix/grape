from fastai.data.all import *
from fastai.vision.all import *

__all__=['dls_mnist']

def dls_mnist(source:list=None, **kwargs):
  if source is None:
    source = os.path.join(untar_data(URLs.MNIST), 'testing')
  db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220922), **kwargs)