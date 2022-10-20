from fastai.data.all import *
from fastai.vision.all import *

from .misc import get_annotations_voc
from .transforms import PointScalerReverse

__all__=['dls_mnist', 'dls_voc', 'dls_voc_tiny']

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
    source = os.path.join(untar_data(URLs.MNIST), 'testing')
  db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220922), **kwargs)


#--------------------------------------------------------------------
# # dataloaders on VOC dataset
def get_voc_vocab():
  return ['aeroplane',   'bicycle',  'bird',    'boat',      'bottle',
          'bus',         'car',      'cat',     'chair',     'cow',
          'diningtable', 'dog',      'horse',   'motorbike', 'person',
          'pottedplant', 'sheep',    'sofa',    'train',     'tvmonitor']

def _get_voc(source:Any):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [{'image_file': img, 'bboxes': bbox, 'labels': label} for img, bbox, label in zip(images, lbls, bboxes)]

def _duple(x):
  return x * 8

def _unscale_tensorbbox(x:TensorBBox):
  sz = x.img_size
  
def _debug_break(x):
  return x

def dls_voc(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/VOC/VOC2007/VOCdevkit/VOC2007')
  valid_pct = getattr(kwargs, 'valid_pct', 0.1)
  seed = getattr(kwargs, 'seed', 20220927)
  db = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
                 get_items=_get_voc,
                 splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
                 getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
                 n_inp=1)
  ds = db.datasets(source)  
  return ds.dataloaders(bs=1, num_workers=0,
                        after_item=[PointScaler(), Resize(304, ResizeMethod.Pad, PadMode.Zeros), ToTensor()],
                        before_batch=[_duple],
                        after_batch=[IntToFloatTensor(), *aug_transforms(), PointScalerReverse(order=90)],
                        **kwargs)
  
def dls_voc_tiny(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_tiny')
  return dls_voc(source, **kwargs)

def dls_voc_test(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_test')
  return dls_voc(source, **kwargs)
