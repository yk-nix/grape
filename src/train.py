import platform
import os

from fastai.vision.all import *

from lib.models.all import *
from lib.exfastai.all import *


def dls_test(dls):
  train_dl, valid_dl = dls_voc()
  for i, x in enumerate(train_dl):
    print(i)
  for i, x in enumerate(valid_dl):
    print(i)

def voc_train():
  dls = dls_voc(source=Path('data/VOC/VOC2007/VOCdevkit/VOC2007'),
                list_dir=os.path.join('ImageSets', 'Layout'),
                after_item=[ToTensor(), PointScaler()],
                before_batch=ExpandBatch(times=8),
                after_batch=[IntToFloatTensor(), 
                  *aug_transforms(size=304,
                                  do_flip=True,
                                  flip_vert=False,
                                  max_rotate=10,
                                  max_lighting=0.2,
                                  min_zoom=0.8,
                                  max_zoom=1.0,
                                  max_warp=0.1,
                                  pad_mode=PadMode.Zeros), 
                  PointScalerReverse(order=90)])  
  model = SSD(num_classes=len(dls.vocab),
              backbone = 'vgg16_bn',
              batch_normal = True,
              image_size = 304,
              anchor_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
              anchor_size_limits = [(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)],
              feature_shapes = [(38,38), (19,19), (10,10), (5,5), (3,3), (1,1)],
              feature_scales = [(8,8), (16,16), (32,32), (64,64), (100,100), (304,304)],
              pretrained=False)
  loss_func = SSDLossFlat(ssd=model, 
                          negative_sample_rate = 0.5,
                          encode_variance = [1.0, 1.0],
                          match_iou_threshold = 0.7)
  opt_func = partial(SGD, mom=0.9)
  learn = create_learner(name = 'voc',
                         dls = dls,
                         model = model,
                         cbs = None,
                         loss_func = loss_func,
                         opt_func = opt_func)
  learn.fit(20, start_epoch=0)
  
if __name__ == '__main__':
  sys = platform.system()
  cwd = os.path.abspath('.')
  if sys == 'Linux' and  cwd != '/home/yoka/grape':
    raise ValueError(f'Wrong working directory: {cwd}')
  voc_train()
