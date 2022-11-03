from fastai.data.all import *
from fastai.vision.all import *
from matplotlib.hatch import VerticalHatch

from .misc import get_annotations_voc
from .transforms import PointScalerReverse, ExpandBatch

__all__=['dls_mnist', 'dls_voc', 'dls_voc_tiny', 'dls_voc_test', 'get_voc_items']


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
  
def get_voc_items(source):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [{'image_file': img, 'bboxes': bbox, 'labels': label} for img, bbox, label in zip(images, lbls, bboxes)]

@delegates(FilteredBase.dataloaders)
def _dls_voc(items, valid_pct, seed, bs, num_workers, image_size, **kwargs):
  splits = ([90], [0])
  #splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(items)
  datasets = Datasets(items=items,  splits=splits, n_inp=1,
                      tfms=[lambda x: PILImage.create(x['image_file']),
                            lambda x: TensorBBox.create(x['bboxes']),
                            [lambda x: x['labels'], MultiCategorize(add_na=True, vocab=_voc_vocab)]])
  return datasets.dataloaders(bs=bs, num_workers=num_workers, before_batch=ExpandBatch(times=8),
                              after_item=[ToTensor(), PointScaler()],
                              after_batch=[IntToFloatTensor(), 
                                           *aug_transforms(size=image_size, min_zoom=0.8, max_zoom=1.0, max_warp=0.1, pad_mode=PadMode.Zeros), 
                                           PointScalerReverse(order=90)],
                              **kwargs)
  # return datasets.dataloaders(bs=bs, num_workers=num_workers, before_batch=ExpandBatch(times=1),
  #                             after_item=[PointScaler(),
  #                                         Resize(image_size, ResizeMethod.Pad, PadMode.Zeros),
  #                                         ToTensor()],
  #                             after_batch=[IntToFloatTensor(), 
  #                                          *aug_transforms(max_rotate=0., max_warp=0., max_lighting=0., max_zoom=1., do_flip=True, flip_vert=True),
  #                                          PointScalerReverse(order=90)],
  #                             **kwargs)

def dls_voc(source:Any=None, valid_pct=0.1, seed=20220927, bs=1, num_workers=0, image_size=304, **kwargs):
  if source is None:
    source = Path('data/VOC/VOC2007/VOCdevkit/VOC2007')
  items = get_voc_items(source)
  return _dls_voc(items, valid_pct, seed, bs, num_workers, image_size, **kwargs)
  
def dls_voc_tiny(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_tiny')
  return dls_voc(source, **kwargs)

def dls_voc_test(source:Any=None, **kwargs):
  if source is None:
    source = Path('data/voc_test')
  return dls_voc(source, valid_pct=.99, **kwargs)
