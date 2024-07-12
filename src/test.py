import numpy as np
import matplotlib.pyplot as plt
import time
import socket
import torch
import PIL

from lib.uitls.file import join_path_file
from lib.datasets import voc
from lib.uitls.vision import show_images, unnormalize


from typing import Any
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data.dataloader import DataLoader
from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights

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

# data_root = 'F:/data'
# transforms = voc.voc_detection_transforms_wrapper(v2.Compose([
#   v2.PILToTensor(),
#   v2.ToDtype(torch.float, scale=True)
#   # v2.Resize((224, 224)),
#   # # v2.Lambda(tf_print),
#   # v2.RandomRotation((30, 30))
# ]))
# dataset = datasets.VOCDetection(data_root, image_set = 'trainval', transforms = transforms)
# dataloader = DataLoader(dataset, 8, collate_fn=lambda x: x)
# size = (224, 224)
# image_mean = [0.48235, 0.45882, 0.40784]
# image_std = [1.0 / 255.0] * 3
# transform = GeneralizedRCNNTransform(min_size=min(size), max_size=max(size), size_divisible=1, fixed_size=size, image_mean=image_mean, image_std=image_std)
# for e in dataloader:
#   x, y = zip(*e)
#   x, y = transform(x, y)
#   print(x.image_sizes)
#   show_images(unnormalize(x.tensors, image_mean, image_std).unbind(0))
#   break


weights = SSD300_VGG16_Weights.DEFAULT
ssd = ssd300_vgg16(weights = weights, 
                   score_thresh = 0.4,
                   trainable_backbone_layers = 0,
                   num_classes = 2)
ssd.name = 'ssd300_vgg16_ped'
ssd.version = ""

# ssd = ssd300_vgg16()
print(ssd)
ssd.eval()
image = PIL.Image.open('F:/data/test/FudanPed00074.png')
tf = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float, scale=True)])
x = tf(image)
with torch.no_grad():
  y = ssd([x])
  scores = y[0]['scores']
  # boxes = y[0]['boxes'].to(torch.int)[scores > threshold]
  # labels = [weights.meta['categories'][i] for i in y[0]['labels'][scores > threshold].flatten().tolist()]
  boxes = y[0]['boxes'].to(torch.int)
  labels = [weights.meta['categories'][i] for i in y[0]['labels'].flatten().tolist()]
  show_images([x], [boxes], [labels], width = 2)


# image, annotation = dataset[4]
# boxes = annotation['boxes']
# # print(annotation['boxes'])
# print(boxes)
# labels = [o['name'] for o in annotation['annotation']['object']]
# show_images([image], [boxes], [labels])

