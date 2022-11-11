from fastai.data.all import *
from fastai.vision.all import *
from xml.dom.minidom import parse
from torchvision.ops import batched_nms as nms

__all__ = ['get_annotations_voc', 'post_predict', 'save_as_darknet']

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
  return images, tuple(bbox_labels)

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

#---------------------------------------------------------
## save weight int the format of darknet
def save_as_darknet(weight_file, model, major, minor, revision, iters):
  f = getattr(model, 'to_darknet', None)
  if f is None:
    print(f'{model.name} has not implemented to_darknet.')
    return
  f(weight_file, major, minor, revision, iters)

#---------------------------------------------------------
## select atmost top_n objects whit score > score_threshold
def post_predict(self, _scores, _bboxes, top_n, score_threshold):
  ss, idx = _scores.max(dim=-1)
  ss[idx == 0] = 0
  ss, idx = ss.sort(descending=True)
  top_k = (ss > score_threshold).sum(dim=-1).max().item()
  if top_k < top_n:
    top_n = top_k
  idx = idx.unsqueeze(dim=-1)
  scores = _scores.gather(1, idx.broadcast_to(_scores.shape))[:,:top_n,:]
  bboxes = _bboxes.gather(1, idx.broadcast_to(_bboxes.shape))[:,:top_n,:]   
  scores, labels = scores.max(dim=self.axis)
  label_list, bbox_list, score_list = [], [], []
  for b, s, l in zip(bboxes, scores, labels):
    i = nms(b, s, l, self.nms_iou_threshold)
    label_list.append(l[i])
    bbox_list.append(b[i])
    score_list.append(s[i])
  return torch.stack(bbox_list), torch.stack(label_list), torch.stack(score_list)