from fastai.data.all import *
from fastai.vision.all import *
from xml.dom.minidom import parse

def get_kwarg(key:str, kwargs:dict, default_value:Any=None,  pop:bool=True) -> Any:
  keys = kwargs.keys()
  for k in keys:
    if key.lower() == k.lower():
      value = kwargs[k]
      if pop:
        del kwargs[k]
      return value
  return default_value
  
def get_annotations_voc(path: Any):
  xmls = get_files(path, extensions='.xml', folders=['annotations', 'Annotations'])
  images, bbox_labels = [], []
  for xml in xmls:
    with open(xml) as xml_file:
      doc = parse(xml_file).documentElement
    images.append(doc.getElementsByTagName('filename')[0].childNodes[0].data)
    objs = doc.getElementsByTagName('object')
    bboxes, lbls = [], []
    for obj in objs:
      lbls.append(obj.getElementsByTagName('name')[0].childNodes[0].data)
      xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
      ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
      xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
      ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
      bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
    bbox_labels.append((bboxes, lbls))
  return images, bbox_labels

def get_coco_tiny(source):
  annotation_file = os.path.join(source, 'train.json')
  images, lbl_bboxes = get_annotations(annotation_file, prefix=os.path.join(source, 'train\\'))
  lbls, bboxes = zip(*lbl_bboxes)
  return [list(o) for o in zip(images, lbls, bboxes)]

def get_voc(source):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [list(o) for o in zip(images, lbls, bboxes)]

def point_scale_boxes(boxes:Tensor,  # shape: n x 4
                      image_size):
    scaler = PointScaler()
    scaler.sz = image_size
    _boxes = []
    for i in range(len(boxes)):
        box = boxes[i]
        _boxes.append(scaler(TensorBBox.create(box, img_size=image_size)))
    return torch.cat(_boxes)