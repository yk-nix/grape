from fastai.data.all import *
from fastai.vision.all import *
from xml.dom.minidom import parse

from .transforms import PointScalerReverse, ExpandBatch

__all__ = ['VOCDatasets']

@delegates(Datasets.__init__)
class VOCDatasets(Datasets):
  _vocab = 'aeroplane bicycle bird boat bottle bus car cat chair cow \
            diningtable dog horse motorbike person pottedplant sheep \
            sofa train tvmonitor'.split()
  train_file, valid_file, test_file = 'train.txt', 'val.txt', 'test.txt'
  
  @classmethod
  def get_source_path(cls, xml_path):
    return Path(xml_path).parent.parent
  
  @classmethod
  def get_items(cls, source: Any):
    source = Path(source)
    items = []
    if os.path.exists(source):
      if os.path.isdir(source):
        xmls = get_files(source, extensions='.xml', folders=['annotations', 'Annotations'])
        for xml in xmls:
          items.append(cls.get_item(xml)) 
      else:
        items, _ =  cls.get_items_from_file (source)   
    return items
  
  @classmethod
  def get_item(cls, xml_path:str):
    with open(xml_path) as xml:
      doc = parse(xml).documentElement
    image_name = doc.getElementsByTagName('filename')[0].childNodes[0].data
    image_file = os.path.join(cls.get_source_path(xml_path), 'JPEGImages', image_name)
    objs = doc.getElementsByTagName('object')
    bboxes, lbls = [], []
    for obj in objs:
      lbls.append(obj.getElementsByTagName('name')[0].childNodes[0].data)
      xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
      ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
      xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
      ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
      bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    return {'image_file': image_file,
            'bboxes': bboxes,
            'labels': lbls}
    
  @classmethod
  def get_items_from_file(cls, source):
    source, items, lines, count = Path(source), [], [], 0
    with open(source, 'r') as f:
      for line in f.readlines():
        line = line.strip()
        if line:
          items = items + [cls.get_item(os.path.join(source.parent.parent.parent, 'Annotations', line + ".xml"))]
          count += 1
    return items, count
  
  def __init__(self, source='', list_dir=None, valid_pct=0.2, seed=20221103, tfms=None, add_na=True, **kwargs):
    self._kwargs = kwargs       
    self.source = Path(source)
    items, splits = self._get_items(list_dir, valid_pct, seed)
    tfms = [lambda x: PILImage.create(x['image_file']),
            lambda x: TensorBBox.create(x['bboxes']),
            [lambda x: x['labels'], MultiCategorize(add_na=add_na, vocab=self._vocab)]] if tfms is None else tfms
    n_inp = self._kwargs.pop('n_inp', 1)
    tfms = self._kwargs.pop('tfms', tfms)
    super().__init__(items=items, tfms=tfms, n_inp=n_inp, splits=splits, **self._kwargs)
  
  def __get_items(self, list_dir):
    items, splits, start = [], [], 0
    self.train_file = os.path.join(self.source, list_dir, self.train_file)
    self.valid_file = os.path.join(self.source, list_dir, self.valid_file)
    self.test_file = os.path.join(self.source, list_dir, self.test_file)
    for file in (self.train_file, self.valid_file, self.test_file):
      if os.path.exists(file):
        lines, count = [], 0
        with open(file, 'r') as f:
          lines = f.readlines()
        for line in lines:
          line = line.strip()
          if len(line) > 0:
            items = items + [self.get_item(os.path.join(self.source, 'Annotations', line + ".xml"))]
            count += 1
        splits.append(range(start, start + count))
        start += count
    return items, splits
  
  def _get_items(self, list_dir, valid_pct, seed,):
    items, splits, start = self._kwargs.pop('items', []), self._kwargs.pop('splits', []), 0
    _items = [{'image_file':torch.zeros(1,1,3, dtype=torch.uint8), 'bboxes':[[0,0,0,0]], 'labels':['#na#']}]
    _splits = [[0], [0]]
    if len(items) == 0:
      if list_dir is not None:
        items, splits = self.__get_items(list_dir)
      else:
        items = self.get_items(self.source)
    else:
      if len(splits) == 0:
        splits = [list(range(len(items)))]
    if len(splits) == 0 and len(items) > 0:
      splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(items)
    if len(splits) == 0:
      splits = _splits
    if len(items) == 0:
      items, splits = _items, _splits
    return items, splits
  
  def subset(self, i):
    return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp)
  
  def new_empty(self): 
    return type(self)(tls=[tl.new_empty() for tl in self.tls], n_inp=self.n_inp)
  
  @delegates(Datasets.dataloaders)
  def dataloaders(self,
                  image_size=304, bs=8, num_workers=0,
                  do_flip=True, flip_vert=False, max_rotate=10, max_lighting=0.2, min_zoom=0.8, max_zoom=1.0, max_warp=0.1, pad_mode=PadMode.Zeros,
                  after_item=None, before_batch=None, after_batch=None, **kwargs):
    after_item = [ToTensor(), PointScaler()] if after_item is None else after_item
    before_batch = ExpandBatch(times=bs) if before_batch is None else before_batch
    after_batch = [IntToFloatTensor(), 
                  *aug_transforms(size=image_size,
                                  do_flip=do_flip,
                                  flip_vert=flip_vert,
                                  max_rotate=max_rotate,
                                  max_lighting=max_lighting,
                                  min_zoom=min_zoom,
                                  max_zoom=max_zoom,
                                  max_warp=max_warp,
                                  pad_mode=pad_mode), 
                  PointScalerReverse(order=90)] if after_batch is None else after_batch
    return super().dataloaders(bs=1, num_workers = num_workers, after_item=after_item, before_batch=before_batch, after_batch=after_batch)