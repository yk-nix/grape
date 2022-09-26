from fastai.data.all import *
from fastai.test_utils  import *
from fastai.callback.all import *
from fastai.learner import *
from fastai.vision.all import *
from pandas import DataFrame
from pyparsing import dblSlashComment
from sklearn.model_selection import train_test_split
import posixpath

from lib.utils.callbacks import AutoPlotCallback, AutoSaveCallback
from lib.utils.dataloaders import dls_mnist
from lib.utils.learners import create_learner
from lib.models.lenet import LeNet



def train_test():
  dls = dls_mnist(num_workers=0)
  model = LeNet()
  cbs = AutoSaveCallback()
  learn = create_learner(dls, model, cbs, 'test')
  learn.fit(10)


def load_test():
  cb = AutoSaveCallback()
  p = Path('models/test/lenet/metrics')
  cb.load('001', p)
  

if __name__ == '__main__':
  load_test()



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
# db = coco_tiny
# ds = db.datasets(source)

# voc = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
#                 get_items=get_voc,
#                 splitter=RandomSplitter(valid_pct=0.2, seed=20220815),
#                 getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
#                 n_inp=1)
# source = 'D:\grapefruit\data\VOC\VOCdevkit\VOC2007'

# db = coco_tiny
# ds = db.datasets(source)
# img, bbox, lbl = ds[23]
# print(bbox)
# resizer = Pipeline([PointScaler(do_scale=False), Resize((256, 256), ResizeMethod.Pad, pad_mode=PadMode.Zeros)])
# #resizer = Resize(256, ResizeMethod.Pad, pad_mode=PadMode.Zeros)
# img = resizer(img)
# bbox = resizer(bbox)
# print(bbox)
# LabeledBBox(bbox, ds.vocab[lbl]).show(ctx=img.show())
# plt.show()

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

# path = untar_data(URLs.CAMVID_TINY)
# dls = SegmentationDataLoaders.from_label_func(
#     path, bs=8, fnames = get_image_files(path/"images"),
#     label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
#     codes = np.loadtxt(path/'codes.txt', dtype=str),
#     num_workers=0
# )
# def main():
#   learn = unet_learner(dls, resnet34)
#   learn.fine_tune(8)
#   learn.show_results(max_n=6, figsize=(7,8))

# if __name__ == '__main__':
#   main()
