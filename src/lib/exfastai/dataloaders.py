from fastai.data.all import *
from fastai.vision.all import *

from .misc import get_annotations_voc

__all__=['dls_mnist', 'dls_voc']


## dataloaders on MNIST dataset
def dls_mnist(source:list=None, **kwargs):
  if source is None:
    source = os.path.join(untar_data(URLs.MNIST), 'testing')
  db = DataBlock([ImageBlock(PILImageBW), CategoryBlock], get_items=get_image_files, get_y=parent_label)
  return db.dataloaders(source, bs=8, splitter=RandomSplitter(valid_pct=0.3, seed=20220922), **kwargs)



## dataloaders on VOC dataset
def _get_voc(source:Any):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [list(o) for o in zip(images, lbls, bboxes)]

def _duple(x):
  return x * 8

def dls_voc(source:Any=None):
  if source is None:
    source = Path('data/VOC/VOC2007/VOCdevkit/VOC2007')
  db = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
                 get_items=_get_voc,
                 splitter=RandomSplitter(valid_pct=0.2, seed=20220927),
                 getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
                 n_inp=1)
  ds = db.datasets(source)
  return ds.dataloaders(bs=1, num_workers=0,
                        after_item=[BBoxLabeler(), PointScaler(), ToTensor(), Resize(304, ResizeMethod.Pad, PadMode.Zeros)],
                        before_batch=[_duple],
                        after_batch=[IntToFloatTensor(), *aug_transforms()])