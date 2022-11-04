from fastai.data.all import *
from fastai.vision.all import *

from xml.dom.minidom import parse

class VOCDatasets(Datasets):
  _vocab = 'aeroplane bicycle bird boat bottle bus car cat chair cow \
            diningtable dog horse motorbike person pottedplant sheep \
            sofa train tvmonitor'.split()
  @classmethod
  def get_items(cls, source: Any):
    xmls = get_files(source, extensions='.xml', folders=['annotations', 'Annotations'])
    items = []
    for xml in xmls:
      items.append(cls.get_item(source, xml))
    return items
  
  @classmethod
  def get_item(cls, source:Any, xml_path:str):
    with open(xml_path) as xml:
      doc = parse(xml).documentElement
    image_name = doc.getElementsByTagName('filename')[0].childNodes[0].data
    image_file = os.path.join(source, 'JPEGImages', image_name)
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
  
  
  def __init__(self, source, list_dir='ImageSets/Layout', valid_pct=0.2, seed=20221103, **kwargs):
    self.source = source
    items = []
    tfms = [lambda x: PILImage.create(x['image_file']),
            lambda x: TensorBBox.create(x['bboxes']),
            [lambda x: x['labels'], MultiCategorize(add_na=True, vocab=self._vocab)]]
    n_inp = 1
    if list_dir is None:
      items = self.get_items(source)
      splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(items)
    else:
      splits, start = [], 0
      train_file = os.join(source, list_dir, 'train.txt')
      valid_file = os.join(source, list_dir, 'valid.txt')
      test_file = os.join(source, list_dir, 'test.txt')
      for file in (train_file, valid_file, test_file):
        if os.path.exists(file):
          lines = []
          with open(file, 'r') as f:
            lines = f.readlines()
          if len(lines) > 0:
            items.append([self.get_item(source, line) for line in lines])
            splits.append(range(start, start + len(lines)))
            start = start + len(lines)
    super().__init__(items=items, tfms=tfms, n_inp=n_inp, splits=splits)
    
    