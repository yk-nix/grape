from typing import NoReturn
from torchvision import datasets
from torchvision.datasets import VOCDetection
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from threading import Thread

from learner import Learner
from lib.datasets.voc import voc_detection_transforms_wrapper, voc_detection_collate_fn
from lib.models.lenet import LeNet

import socket
import torch
import torchvision.transforms.v2 as v2

def controller(learner: Learner) -> Thread :
  def closure():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 12345))
    while True:
      data, _ = sock.recvfrom(2048)
      cmd = data.decode().strip(' \t\r\n')
      if cmd == 'stop' or cmd == 'exit':
        if learner is not None:
          learner.stop()
        break
    sock.close()
    print('stop controller')
  return Thread(target = closure)

def logger(learner: Learner, freq: int = 200) -> NoReturn:
  if learner.loop % freq == 0:
    info = {'dataloader': 0, 'forward': 0, 'backward': 0, 'update': 0, 'loss': 0}
    for log in learner.logs:
      info['dataloader'] += log['dataloader']
      info['forward'] += log['forward']
      info['backward'] += log['backward']
      info['update'] += log['update']
      info['loss'] += log['loss']
    learner.logs.clear()
    info['loss'] /= freq
    print(info)

def weight_saver(learner: Learner) -> NoReturn:
  learner.save()
  values = learner.eval()
  confidences, predictions, labels = tuple(zip(*values))
  confidences = torch.hstack(confidences)
  predictions = torch.hstack(predictions)
  labels = torch.hstack(labels)
  print(f'------- epoch={learner.epoch} accuracy rate: {torch.sum(predictions == labels).item() / len(labels) * 100}% --------')

def main() -> NoReturn:
  data_root = 'F:/data'
  weight_root = 'F:/weight'
  transforms = v2.Compose([ v2.PILToTensor(), v2.ToDtype(torch.float, scale = True)])
  train_dataset = datasets.MNIST(root = data_root, train = True, transform = transforms)
  eval_dataset = datasets.MNIST(root = data_root, train = False, transform = transforms)
  train_dataloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True, num_workers = 0)
  eval_dataloader = DataLoader(dataset = eval_dataset, batch_size = 64, shuffle = True, num_workers = 0)
  model = LeNet(10)
  loss_fn = CrossEntropyLoss()
  optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
  lr_scheduler = StepLR(optimizer, step_size = 10, gamma = 0.8)
  learner = Learner(root = weight_root,
                    name = model.name + "_" + model.version,
                    train_dataloader = train_dataloader,
                    eval_dataloader = eval_dataloader,
                    model = model,
                    loss_fn = loss_fn,
                    eval_fn = LeNet.eval,
                    optimizer = optimizer,
                    lr_scheduler= lr_scheduler,
                    loop_cb = lambda learner: logger(learner = learner),
                    epoch_cb = lambda learner: weight_saver(learner = learner))
  learner.load('00130382.pth')
  # controller(learner).start()
  learner.train()
  

if __name__ == '__main__':
  main()