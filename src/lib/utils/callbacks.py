from fastai.data.all import *
from fastai.callback.all import *

__all__=['AutoPlotCallback', 'AutoSaveCallback']


@patch
def save(self:Callback, file, metric_dir='metrics', ext='.pkl'):
  if self.learn:
    file = join_path_file(file, self.learn.path/self.learn.model_dir/metric_dir, ext)
  else:
    file = join_path_file(file, metric_dir, ext)
  with open(file, 'wb') as f:
    pickle.dump(self, f)

@patch
def load(self:Callback, file, metric_dir='metrics', ext='.pkl'):
  if self.learn:
    file = join_path_file(file, self.learn.path/self.learn.model_dir/metric_dir, ext)
  else:
    file = join_path_file(file, metric_dir, ext)
  with open(file) as f:
    cb = pickle.load(f)
    for attr in cb._stateattrs:
      self.attr = cb.attr

class AutoPlotCallback(Callback):
  order = 90
  def before_epoch(self):
    plt.ion()
  
  def after_batch(self):
    plt.clf()
    self.learn.recorder.plot_loss()
    plt.pause(0.01)

  def after_fit(self):
    plt.ioff()

class AutoSaveCallback(Callback):
  order = 99    
  def after_epoch(self):
    file = f'{self.learn.epoch + 1:03}'
    self.learn.save(file)
    if self.learn.recorder:
      self.learn.recorder.save(file)