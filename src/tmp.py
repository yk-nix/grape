
































from fastai.data.all import *
from fastai.test_utils  import *
from fastai.callback.all import *
from fastai.learner import *
from fastai.vision.all import *
from pandas import DataFrame
from pyparsing import dblSlashComment
from pyrsistent import v
from sklearn.model_selection import train_test_split
import posixpath

from lib.utils.misc import get_annotations_voc


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

# mnist = DataBlock(blocks=[ImageBlock(PILImageBW), CategoryBlock()],
#                   get_items=ImageGetter(folders=['training']),
#                   splitter=RandomSplitter(valid_pct=0.3, seed=20220815),
#                   get_y=parent_label)

# mnist_tiny = DataBlock(blocks=[ImageBlock(PILImageBW), CategoryBlock()],
#                        get_items=ImageGetter(folders=['train', 'valid']),
#                        splitter=GrandparentSplitter(),
#                        get_y=parent_label)

# def create_bbox(x):
#   return TensorBBox.create(x[1], img_size=Image.open(x[0]).size)

# coco_tiny = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
#                       get_items=get_coco_tiny,
#                       splitter=RandomSplitter(valid_pct=0.2, seed=20220815),
#                       getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
#                       n_inp=1)
# source = untar_data(URLs.COCO_TINY)

# voc = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
#                 get_items=get_voc,
#                 splitter=RandomSplitter(valid_pct=0.2, seed=20220815),
#                 getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
#                 n_inp=1)
# source = 'D:\grapefruit\data\VOC\VOCdevkit\VOC2007'

# db = voc
# ds = db.datasets(source)

# def _duple(x):
#   return x * 4

# train_dl, valid_dl = ds.dataloaders(bs=1, num_workers=0,
#                                     after_item=[BBoxLabeler(), PointScaler(), ToTensor(), Resize(512, ResizeMethod.Pad)],
#                                     before_batch=[_duple],
#                                     after_batch=[IntToFloatTensor(), *aug_transforms()])
# for i, b in enumerate(train_dl):
#   print(b[1].img_size, b[0].shape)
#   train_dl.show_batch(b)
#   plt.show()
TensorImage
file_name = r'D:\grapefruit\data\test\chelsea.png'

img = PILImage.create(file_name)
pass