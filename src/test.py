import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import torch
import PIL
import re
import configparser
import os
import torchvision.transforms.v2.functional as tf


from lib.uitls.file import join_path_file
from lib.datasets import voc
from lib.uitls.vision import show_images, unnormalize, show_image
from lib.datasets.pede import PedDetection
from lib.learners.pedestrain import PedestrainDetector
from lib.learners.learner import Learner

from typing import Any, Union, Dict, NoReturn, List
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.datasets.imagenet import ImageNet
from torchvision.tv_tensors import BoundingBoxes
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data.dataloader import DataLoader
from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights
from threading import Thread
from pathlib import Path
from PIL import Image

# weight_root = 'F:/weight'
# print(join_path_file(weight_root, 'lenet5', file = ''))


# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# # s.bind(('0.0.0.0', 2412))
# # while True:
# #   data, _ = s.recvfrom(2048)
# #   print(data.decode())
# count = 0
# while True:
#   count += 1
#   s.sendto(('hello, cout = ' + str(count)).encode(), ('192.168.3.241', 2412))
#   time.sleep(1)
#   print(count)

# fig = plt.figure()
# ax = fig.subplots()

# t = np.linspace(0, 10, 100)
# y = np.sin(t)
# ax.axis([0, 10, 0, 2])
# ax.set_aspect(3)

# while True:
#   ax.plot(t, y)
#   plt.pause(0.5)
#   ax.cla()
#   t += np.pi / 30
#   y = np.sin(t)

def tf_print(input: Any) -> Any:
  print(input)
  return input

root = 'F:/data'

# dataset = datasets.VOCDetection(root, image_set='train', transform=v2.functional.pil_to_tensor)
# dataset.images = [re.compile(' .*\.').sub('.', e) for e in dataset.images]
# dataset.targets = [re.compile(' .*\.').sub('.', e) for e in dataset.targets]
# for img, target in dataset:
#   a = target['annotation']
#   boxes = torch.tensor([[int(v) for v in o['bndbox'].values()] for o in a['object']])
#   labels = [o['name'] for o in a['object']]
#   print(a['filename'])
#   show_images([img], [boxes], [labels])

# dataset = datasets.ImageFolder(root + '/' + 'test')
# print(dataset[0])

# dataset = PedDetection(root, transform=v2.functional.pil_to_tensor)
# for img, target in dataset:
#   boxes = torch.tensor(target['bndboxes'])
#   show_images([img], [boxes])



# transforms = voc.voc_detection_transforms_wrapper(v2.Compose([
#   v2.PILToTensor(),
#   v2.ToDtype(torch.float, scale=True)
#   # v2.Resize((224, 224)),
#   # # v2.Lambda(tf_print),
#   # v2.RandomRotation((30, 30))
# ]))
# dataset = datasets.VOCDetection(root, image_set = 'trainval', transforms = transforms)
# dataloader = DataLoader(dataset, 8, collate_fn=lambda x: x)
# size = (224, 224)
# image_mean = [0.48235, 0.45882, 0.40784]
# image_std = [1.0 / 255.0] * 3
# transform = GeneralizedRCNNTransform(min_size=min(size), max_size=max(size), size_divisible=1, fixed_size=size, image_mean=image_mean, image_std=image_std)
# for e in dataloader:
#   x, y = zip(*e)
#   x, y = transform(x, y)
#   print(x.image_sizes)
#   boxes = [e['boxes'] for e in y]
#   show_images(unnormalize(x.tensors, image_mean, image_std).unbind(0), boxes)
#   break


# weights = SSD300_VGG16_Weights.DEFAULT
# ssd = ssd300_vgg16(weights = weights, 
#                    score_thresh = 0.4,
#                    trainable_backbone_layers = 0,
#                    num_classes = 2)
# # ssd = ssd300_vgg16()
# print(ssd)
# ssd.eval()
# image = PIL.Image.open('F:/data/test/FudanPed00074.png')
# tf = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float, scale=True)])
# x = tf(image)
# with torch.no_grad():
#   y = ssd([x])
#   scores = y[0]['scores']
#   # boxes = y[0]['boxes'].to(torch.int)[scores > threshold]
#   # labels = [weights.meta['categories'][i] for i in y[0]['labels'][scores > threshold].flatten().tolist()]
#   boxes = y[0]['boxes'].to(torch.int)
#   labels = [weights.meta['categories'][i] for i in y[0]['labels'].flatten().tolist()]
#   show_images([x], [boxes], [labels], width = 2)


# image, annotation = dataset[4]
# boxes = annotation['boxes']
# # print(annotation['boxes'])
# print(boxes)
# labels = [o['name'] for o in annotation['annotation']['object']]
# show_images([image], [boxes], [labels])


def load_dicts(file: Union[str, Path]) -> List[Dict]:
  info = []
  with open(file, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if len(line) == 0:
        continue
      info.append(eval(line))
  return info
    

if __name__ == '__main__':
  # losses = [e['loss'] for e in load_dicts('tmp/log.txt') if e['loss'] < 0.01]
  # plt.plot(losses)
  # plt.show()

  # detector = PedestrainDetector()
  # for x, y in detector.learner.train_dataloader:
  #   boxes = [e['boxes'] for e in y]
  #   show_images(x, boxes)

  # transforms = v2.Compose([v2.PILToTensor(),
  #                          v2.RandomHorizontalFlip(),
  #                         #  v2.RandomVerticalFlip(),
  #                          v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.3),
  #                          v2.RandomGrayscale(),
  #                          v2.ToDtype(torch.float, scale = True)])
  # dataset = PedDetection(root, image_set='train', transforms = transforms)
  # for x, y in dataset:
  #   show_images([x], [y['boxes']])

  # img = tf.pil_to_tensor(Image.open('F:/data/test/astronaut.png'))
  # transforms = v2.Compose([
  #   v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.3),
  #   v2.RandomPosterize(6, 0.2),
  #   v2.RandomGrayscale(),
  # ])
  # show_images([img] + [transforms(img) for _ in range(7)])
  
  dataset = ImageNet(root = 'F:/data/ILSVRC2012/ILSVRC2012')
  print(dataset[0])
  # fpath = 'F:/data/ILSVRC2012/ILSVRC2012\\ILSVRC2012_devkit_t12.tar.gz'
  #         'F:\data\ILSVRC2012\ILSVRC2012\ILSVRC2012_devkit_t12.tar'

  # print(os.path.isfile(fpath))
  pass