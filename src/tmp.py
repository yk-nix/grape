
































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

metric_dir = "Y:\metrics"
files = L(os.listdir(metric_dir)).filter(lambda fn: fn.startswith('cls_mnist'))
import re

for file in files:
  print(file)
  seo = re.search(r'(.*)_(\d+)-(\d+).meta', file)
  if seo:
    start_epoch = int(seo.group(2))
    end_epoch = int(seo.group(3))
    print(start_epoch, end_epoch)
    

# def save_recorder(file: str, recorder: Recorder, recorder_dir="recorder"):
#   data = {
#     'lrs': recorder.lrs,
#     'iters': recorder.iters,
#     'losses': recorder.losses,
#     'loss': recorder.loss,
#     'smooth_loss': recorder.smooth_loss,
#     'add_time': recorder.add_time,
#     'metrics': recorder.metrics,
#     'train_metrics': recorder.train_metrics,
#     'valid_metrics': recorder.valid_metrics,
#     'metric_names': recorder.metric_names,
#     'log': recorder.log
#   }
#   torch.save(data, os.path.join(recorder.learn.path, recorder_dir, f'{file}.rec'))

# def store_recorder(file: str, Recorder: Recorder):
#   pass

# class AutoSaveCallback(Callback):
#   order = 99
#   def after_epoch(self):
#     self.learn.save(f'{self.learn.epoch}')
#     save_recorder(f'{self.learn.epoch}', self.learn.recorder)

# class AutoPlotCallback(Callback):
#   order = 90
#   def before_epoch(self):
#     plt.ion()
  
#   def after_batch(self):
#     plt.clf()
#     self.learn.recorder.plot_loss()
#     plt.pause(0.01)

#   def after_fit(self):
#     plt.ioff()
    
# def main():
#   learn = digit_classifier()
#   learn.load('9')
#   learn.add_cbs([AutoSaveCallback()])
#   learn.show_training_loop()
#   learn.fit(20, start_epoch=10)

# if __name__ == '__main__':
#   main()


# # class _IntFloatTfm(Transform):
# #     def encodes(self, o):  return TitledInt(o)
# #     def decodes(self, o):  return TitledFloat(o)
# # int2f_tfm=_IntFloatTfm()

# # def _neg(o): return -o
# # neg_tfm = Transform(_neg, _neg)
# # items = L([1.,2.,3.])

# # class _B(Transform):
# #     def __init__(self): self.m = 0
# #     def encodes(self, o): return o+self.m
# #     def decodes(self, o): return o-self.m
# #     def setups(self, items): 
# #         print(items)
# #         self.m = tensor(items).float().mean().item()

# # class _Cat(Transform):
# #     order = 1
# #     def encodes(self, o):    return int(self.o2i[o])
# #     def decodes(self, o):    return TitledStr(self.vocab[o])
# #     def setups(self, items): 
# #       self.vocab, self.o2i = uniqueify(L(items), sort=True, bidir=True)
  
# # def _lbl(o):
# #   return TitledStr(o.split('_')[0])

# # fns = ['dog_0.jpg','cat_0.jpg','cat_2.jpg','cat_1.jpg','dog_1.jpg']

# # def get_annotations_coco_tiny(source):
# #   annotation_file = os.path.join(source, 'train.json')
# #   images, lbl_bboxes = get_annotations(annotation_file, prefix=os.path.join(source, 'train\\'))
# #   lbls, bboxes = zip(*lbl_bboxes)
# #   return [list(o) for o in zip(images, lbls, bboxes)]

# # mnist = DataBlock(blocks=[ImageBlock(PILImageBW), CategoryBlock()],
# #                   get_items=ImageGetter(folders=['training']),
# #                   splitter=RandomSplitter(valid_pct=0.3, seed=20220815),
# #                   get_y=parent_label)

# # mnist_tiny = DataBlock(blocks=[ImageBlock(PILImageBW), CategoryBlock()],
# #                        get_items=ImageGetter(folders=['train', 'valid']),
# #                        splitter=GrandparentSplitter(),
# #                        get_y=parent_label)

# # coco_tiny = DataBlock(blocks=[ImageBlock(PILImage), BBoxBlock, BBoxLblBlock(add_na=True)],
# #                       get_items=get_annotations_coco_tiny,
# #                       splitter=RandomSplitter(valid_pct=0.2, seed=20220815),
# #                       getters=[lambda o: o[0], lambda o: o[1], lambda o: o[2]],
# #                       n_inp=1)

# # source = untar_data(URLs.COCO_TINY)
# # db = coco_tiny
# # ds = db.datasets(source)
# # train_dl, valid_dl = ds.dataloaders(bs=1, num_workers=0, after_item=[BBoxLabeler(), PointScaler(), ToTensor()])
# # for i, x in enumerate(train_dl):
# #   print(x)
# #   train_dl.show_batch(x)
# #   plt.show()
















# # db = DataBlock(TransformBlock([_Cat(), _lbl]), splitter=IndexSplitter([2,3]))
# # ds = db.datasets(fns)
# # train_dl, valid_dl = ds.dataloaders(bs=2, num_workers=0, drop_last=True)
# # #train_dl, valid_dl = db.dataloaders(fns, bs=2, , num_workers=0, drop_last=True)
# # print(len(train_dl), len(valid_dl))
# # for x in enumerate(train_dl):
# #   print(x)

# ## dataloaders: load dataset and to carry transformations on each data if necessary.

# # mnist = DataBlock(blocks = (ImageBlock(cls=PILImageBW),CategoryBlock),
# #                   get_items = get_image_files,
# #                   splitter = GrandparentSplitter(),
# #                   get_y = parent_label)
# # print(mnist)

# # path = untar_data(URLs.PETS)
# # files = get_image_files(path/'images')

# # def label_func(f):
# #   return f[0].isupper()

# # dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
# # dls.show_batch()
# # plt.show()

# # path = untar_data(URLs.MNIST_TINY)
# # files = ImageGetter(folders=['valid', 'train'])(path)
# # train, valid = GrandparentSplitter()(files)

# # def to_file_name(i: int): return files[i]
# # def open_image(p: Path): return Image.open(p).copy()
# # def img2tensor(img: Image.Image): return TensorImage(array(img)[None])
# # tfms = [[to_file_name, open_image, img2tensor], [to_file_name, parent_label, Categorize()]]

# # train_ds = Datasets(train, tfms)
# # batch_tmfs = [IntToFloatTensor(), Normalize(mean=0, std=1.0)]
# # tdl = TfmdDL(train_ds, bs=4, shuffle=True, after_batch=batch_tmfs)
# # x, y = tdl.one_batch()
# # tdl.show_batch((x, y))
# # print(tfms[1][2].decode(y))
# # print(len(tdl))
# # m = x[0,0,:,:]
# # print(m.shape)
# # print(m.mean(), m.std())
# # print(x.mean(), x.std())
# # plt.show()

# # @docs
# # class Test:
# #   _docs=dict(test='just for test', cls_doc="Test Class")
  
# #   def __init__(self, name:str):
# #     self.name = name
    
# #   def test(self):
# #     print(f"hello {self.name}")




# # print(URLs.BIWI_SAMPLE)
# # print(URLs.CIFAR)
# # print(URLs.COCO_SAMPLE)
# # print(URLs.COCO_TINY)
# # print(URLs.HUMAN_NUMBERS)
# # print(URLs.IMDB)
# # print(URLs.IMDB_SAMPLE)
# # print(URLs.ML_SAMPLE)
# # print(URLs.ML_100k)
# # print(URLs.MNIST_VAR_SIZE_TINY)
# # print(URLs.PLANET_SAMPLE)
# # print(URLs.PLANET_TINY)
# # print(URLs.IMAGENETTE)
# # print(URLs.IMAGENETTE_160)
# # print(URLs.IMAGENETTE_320)
# # print(URLs.IMAGEWOOF)
# # print(URLs.IMAGEWOOF_160)
# # print(URLs.IMAGEWOOF_320)
# # print(URLs.IMAGEWANG)
# # print(URLs.IMAGEWANG_160)
# # print(URLs.IMAGEWANG_320)
# # print(URLs.DOGS)
# # print(URLs.CALTECH_101)
# # print(URLs.CARS)
# # print(URLs.CIFAR_100)
# # print(URLs.CUB_200_2011)
# # print(URLs.FLOWERS)
# # print(URLs.FOOD)
# # print(URLs.MNIST)
# # print(URLs.AMAZON_REVIEWS_POLARITY)
# # print(URLs.DBPEDIA)
# # print(URLs.MT_ENG_FRA)
# # print(URLs.SOGOU_NEWS)
# # print(URLs.WIKITEXT)
# # print(URLs.WIKITEXT_TINY)
# # print(URLs.YAHOO_ANSWERS)
# # print(URLs.YELP_REVIEWS)
# # print(URLs.YELP_REVIEWS_POLARITY)
# # print(URLs.BIWI_HEAD_POSE)
# # print(URLs.CAMVID)
# # print(URLs.CAMVID_TINY)
# # print(URLs.LSUN_BEDROOMS)
# # print(URLs.PASCAL_2007)
# # print(URLs.PASCAL_2012)
# # print(URLs.MACAQUES)
# # print(URLs.ZEBRA_FINCH)
# # print(URLs.TCGA_SMALL)
# # print(URLs.OPENAI_TRANSFORMER)
# # print(URLs.WT103_FWD)
# # print(URLs.WT103_BWD)



