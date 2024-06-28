from typing import NoReturn, Callable, Tuple, Any, Union
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from datetime import datetime
from pathlib import Path

from lib.uitls.file import join_path_file

import os
import torch

class Learner() : 
  '''
    Args: \n
      root (str or ``pathlib.Path``): Root directory of leaner where the weights will be saved
      name (str): the name of this learner, which should be unique
      dataloader (DataLodaer): dataloader
      model (Module): model
      loss_fn (Callable): function to calculate loss
      optimizer (Optimizer): optimizer
      lr_scheduler (LRScheduler): scheduler to update learning rate
      loop_cb (Callable): the callable function has only one parameter which is instance of Learner
      epoch_cb (Callable): same constraints as loop_cb
  '''
  def __init__(self,
               root: Union[str, Path],
               name: str,
               dataloader: DataLoader,
               model: Module,
               loss_fn: Callable,
               optimizer: Optimizer,
               lr_scheduler: LRScheduler = None,
               loop_cb: Callable = None,
               epoch_cb: Callable = None) :
    if isinstance(root, str):
      root = os.path.expanduser(root)
    self.root = root
    self.name = name
    self.dataloader = dataloader
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.loop = 0
    self.epoch = 0
    self.interrupted = False
    self.logs = []
    self.loop_cb = loop_cb
    self.epoch_cb = epoch_cb

  def stop(self) -> NoReturn:
    'stop learner'
    self.interrupted = True

  def forward(self, x: Tensor) -> Tensor:
    'forward + calculate loss'
    return self.model(x)

  def backward(self, loss: Tensor) -> NoReturn:
    '''backward: calculate gradients'''
    loss.backward()

  def update(self) -> NoReturn:
    '''update model paramters'''
    self.optimizer.step()

  def measure_time(self, routine: Callable, *args: Any, returnable: bool = False) -> float:
    '''measure how much time(unit: microsecond) consumed to run this 'routine' '''
    start = datetime.now()
    if returnable:
      ret = routine(*args)
    else:
      routine(*args)
    end = datetime.now()
    if returnable:
      return end.timestamp() - start.timestamp(), ret
    else:
      return end.timestamp() - start.timestamp()
  
  def default_weight_file_name(self) -> str:
    return str.format("{:0>8d}.pth", self.loop)
  
  def default_weight_file_directory(self) -> Path:
    return Path(self.root, self.name)

  def save(self, filename: str = None) -> NoReturn:
    state = {'loop': self.loop,
             'epoch': self.epoch,
             'model': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict()}
    if self.lr_scheduler is not None:
      state['lr_scheduler'] = self.lr_scheduler.state_dict()
    if filename is None:
      filename = self.default_weight_file_name()
    torch.save(state, join_path_file(self.root, self.name, file = filename))

  def load(self, file: Union[str, Path]) -> NoReturn:
    file = Path(file)
    if not os.path.isabs(file):
      file = Path(self.default_weight_file_directory(), file)
    states = torch.load(file)
    self.model.load_state_dict(states['model'])
    self.optimizer.load_state_dict(states['optimizer'])
    if self.lr_scheduler is not None:
      self.lr_scheduler.load_state_dict(states['lr_scheduler'])
    self.loop = states['loop']
    self.epoch = states['epoch']

  def on_exit(self) -> NoReturn:
    self.save()

  def train(self) -> NoReturn:
    while True:
      now = datetime.now()
      for x, y in self.dataloader:
        if self.interrupted:
          self.on_exit()
          return
        self.loop += 1
        log = {'loop': self.loop, 
               'epoch': self.epoch,
               'timestamp': datetime.now,
               'loss': None, 
               'dataloader': datetime.now().timestamp() - now.timestamp()}
        self.optimizer.zero_grad()
        log['forward'], y_hat = self.measure_time(self.forward, x, returnable = True)
        loss = self.loss_fn(y_hat, y)
        log['loss'] = loss.item()
        log['backward'] = self.measure_time(self.backward, loss)
        log['update'] = self.measure_time(self.update)
        self.logs.append(log)
        if self.loop_cb is not None:
          self.loop_cb(self)
        now = datetime.now()
      self.epoch += 1
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
      if self.epoch_cb is not None:
        self.epoch_cb(self)



  