from fastai.data.all import *
from fastai.vision.all import *

from .misc import get_annotations_voc
from .transforms import PointScalerReverse

__all__=['dls_mnist', 'dls_voc', 'dls_voc_tiny']


#--------------------------------------------------------------------
# # dataloaders on MNIST dataset
def dls_mnist(source:list=None, **kwargs):
  if source is None:
    source = os.path.join(untar_data(URLs.MNIST), 'testing')
  db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220922), **kwargs)


#--------------------------------------------------------------------
# # dataloaders on VOC dataset
def _get_voc(source:Any):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [list(o) for o in zip(images, lbls, bboxes)]

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
                        after_item=[BBoxLabeler(), PointScaler(), Resize(304, ResizeMethod.Pad, PadMode.Zeros), ToTensor()],
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
