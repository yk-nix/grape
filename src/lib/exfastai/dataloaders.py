from fastai.data.all import *
from fastai.vision.all import *
from matplotlib.hatch import VerticalHatch

from .misc import get_annotations_voc
from .transforms import PointScalerReverse, ExpandBatch

__all__=['dls_mnist', 'dls_voc', 'dls_voc_tiny', 'dls_voc_test', 'get_voc_items', 'dls_for_detection', 'dls_voc_for_detection']


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
_voc_vocab = 'aeroplane bicycle bird boat bottle bus car cat chair cow \
              diningtable dog horse motorbike person pottedplant sheep \
              sofa train tvmonitor'.split()

_dumpy_items = [{'image_file':torch.zeros(1,1,3, dtype=torch.uint8), 'bboxes':[[0,0,0,0]], 'labels':['#na#']}]

def _dls_for_detection(datasets, image_size, bs=16, num_workers=0,
                      do_flip=True, flip_vert=False, max_rotate=10, max_lighting=0.2, min_zoom=0.8, max_zoom=1.0, max_warp=0.1, pad_mode=PadMode.Zeros,
                      **kwargs):
  return datasets.dataloaders(bs=1,
                              num_workers=num_workers,
                              before_batch=ExpandBatch(times=bs),
                              after_item=[ToTensor(), PointScaler()],
                              after_batch=[IntToFloatTensor(), 
                                           *aug_transforms(size=image_size,
                                                           do_flip=do_flip,
                                                           flip_vert=flip_vert,
                                                           max_rotate=max_rotate,
                                                           max_lightign=max_lighting,
                                                           min_zoom=min_zoom,
                                                           max_zoom=max_zoom,
                                                           max_warp=max_warp,
                                                           pad_mode=pad_mode), 
                                           PointScalerReverse(order=90)],
                              **kwargs)

def get_voc_items(source):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [{'image_file': img, 'bboxes': bbox, 'labels': label} for img, bbox, label in zip(images, lbls, bboxes)]

@delegates(FilteredBase.dataloaders)
def dls_for_detection(items, vocab, image_size, bs=1, num_workers=0, splits=None, random_split=False, valid_pct=0.2, seed=20221103, **kwargs):
  if random_split and splits is None:
    splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(items)
  datasets = Datasets(items=items,  
                      splits=splits, 
                      n_inp=1,
                      tfms=[lambda x: PILImage.create(x['image_file']),
                            lambda x: TensorBBox.create(x['bboxes']),
                            [lambda x: x['labels'], MultiCategorize(add_na=True, vocab=vocab)]])
  return datasets.dataloaders(bs=bs,
                              num_workers=num_workers,
                              before_batch=ExpandBatch(times=8),
                              after_item=[ToTensor(), PointScaler()],
                              after_batch=[IntToFloatTensor(), 
                                           *aug_transforms(size=image_size, min_zoom=0.8, max_zoom=1.0, max_warp=0.1, pad_mode=PadMode.Zeros), 
                                           PointScalerReverse(order=90)],
                              **kwargs)

def dls_voc_for_detection(items=[], image_size=304, **kwargs):
  if len(items) > 0:
    return dls_for_detection(items=items, vocab=_voc_vocab, image_size=image_size, **kwargs)
  else:
    return dls_for_detection(items=_dumpy_items, splits=([0],[0]), vocab=_voc_vocab, image_size=image_size, **kwargs)

def dls_voc(source:Any=None, valid_pct=0.1, seed=20220927, bs=1, num_workers=0, image_size=304, **kwargs):
  if source is None:
    source = Path('data/VOC/VOC2007/VOCdevkit/VOC2007')
  items = get_voc_items(source)
  return dls_voc_for_detection(items=items, image_size=image_size, 
                               bs=bs, num_workers=num_workers,
                               valid_pct=valid_pct, random_splits=True, seed=seed,
                               **kwargs)
  
def dls_voc_tiny(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_tiny')
  return dls_voc(source, **kwargs)

def dls_voc_test(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_test')
  return dls_voc(source, valid_pct=.99, **kwargs)
