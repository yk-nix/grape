from fastai.data.all import *
from fastai.vision.all import *
from matplotlib.hatch import VerticalHatch

from .misc import get_annotations_voc
from .transforms import PointScalerReverse, ExpandBatch
from .datasets import VOCDatasets

__all__=['dls_mnist', 'dls_voc', 'dls_voc_dumy']


#--------------------------------------------------------------------
# override on Datasets
@patch
def show(self:Datasets, o, ctx=None, kwargs_list=None):
  if kwargs_list is None:
    kwargs_list = [dict()] * len(o)
  for o_,tl, kwargs in zip(o,self.tls, kwargs_list): ctx = tl.show(o_, ctx=ctx, **kwargs)
  return ctx

#--------------------------------------------------------------------
# extension-funcitons on Dataloader
@patch
def dump(self:DataLoader):
  print('properties:')
  print(f'  n: {self.n}')
  print(f'  bs: {self.bs}')
  print(f'  shuffle: {self.shuffle}')
  print(f'  indexed: {self.indexed}')
  print(f'  drop_last: {self.drop_last}')
  print(f'  pin_memory: {self.pin_memory}')
  print(f'  timeout: {self.timeout}')
  print(f'  device: {self.device}')
  print(f'  num_workers: {self.num_workers}')
  print(f'  offs: {self.offs}')

#--------------------------------------------------------------------
# # dataloaders on MNIST dataset
def dls_mnist(source:list=None, **kwargs):
  if source is None:
    source = os.path.join(untar_data(URLs.MNIST), 'training')
  db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220922), **kwargs) 

#--------------------------------------------------------------------
# # dataloaders
@delegates(VOCDatasets.dataloaders)
def dls_voc(source:Any=None, list_dir=None, **kwargs):
  if source is None:
    source = Path('data/VOC/VOC2007/VOCdevkit/VOC2007')
  ds = VOCDatasets(source=Path(source), list_dir=list_dir)
  return ds.dataloaders(**kwargs)

@delegates(VOCDatasets.dataloaders)
def dls_voc_dumy(**kwargs):
  return VOCDatasets(items=[]).dataloaders(**kwargs)

