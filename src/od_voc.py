from fastai.data.all import *
from fastai.learner import *
from fastai.vision.all import *

from lib.utils.misc import AutoSaveCallback, get_annotations_voc
from lib.models.ssd import SSD, SSDLoss

__all__ = ['get_dls_voc','voc_detector']


def _get_voc(source:Any):
  images, lbl_bboxes = get_annotations_voc(source)
  lbls, bboxes = zip(*lbl_bboxes)
  images = L(images).map(lambda e: os.path.join(source, 'JPEGImages', e))
  return [list(o) for o in zip(images, lbls, bboxes)]

def _duple(x):
  return x * 8

def get_dls_voc(source:Any=None):
  if source is None:
    source = Path('data/VOC/VOC2012/VOCdevkit/VOC2012')
    #source = Path('D:\grapefruit\data\VOC\VOCdevkit\VOC2007')
  #db = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, TransformBlock(type_tfms=[MultiCategorize(), OneHotEncode()], item_tfms=BBoxLabeler)],
  db = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
                 get_items=_get_voc,
                 splitter=RandomSplitter(valid_pct=0.2, seed=20220830),
                 getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
                 n_inp=1)
  ds = db.datasets(source)
  return ds.dataloaders(bs=1, num_workers=0,
                        after_item=[BBoxLabeler(), PointScaler(do_scale=False), ToTensor(), Resize(304, ResizeMethod.Pad)],
                        before_batch=[_duple],
                        after_batch=[IntToFloatTensor(), *aug_transforms()])
  
@delegates(Learner.__init__)
def voc_detector(data=None, model=None, lr=1e-3, model_name="od_voc", start_epoch=None, **kwargs):
  if data is None: 
    data = get_dls_voc()
  if model is None:
    model = SSD(num_classes=21)
    if start_epoch == 0:
      model.load_pretrained_backbone()
  learn = Learner(data, model, lr=lr, loss_func=SSDLoss(model),
                 opt_func=partial(SGD, mom=0.9), **kwargs)
  if start_epoch and start_epoch > 0:
    weight_file_name = f'{model_name}/{start_epoch:03}'
    learn.load(weight_file_name)
    learn.start_epoch = start_epoch
  else:
    learn.start_epoch = 0
  learn.add_cbs([AutoSaveCallback(model_name=model_name)])
  return learn