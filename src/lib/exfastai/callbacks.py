from fastai.data.all import *
from fastai.callback.all import *
from fastai.learner import *

__all__=['AutoPlotCallback', 'AutoSaveCallback', 'TestSaveCallback']


def _save_recorder(file, recorder: Recorder):
  _states_dict = {
    'lrs': recorder.lrs,
    'losses': recorder.losses,
    'iters': recorder.iters,
    'values': recorder.values,
    'metrics': recorder.metrics,
    'train_metrics': recorder.train_metrics,
    'valid_metrics': recorder.valid_metrics,
    'metric_names': recorder.metric_names,
  }
  torch.save(_states_dict, file)

def _restore_recorder(file, recorder: Recorder, start_epoch):
  _state_dict = torch.load(file)
  iters = _state_dict['iters']
  if start_epoch is None:
    start_epoch = len(iters)
  if start_epoch > 0:
    recorder.last_iter_count = iters[start_epoch - 1]
    recorder.lrs = _state_dict['lrs'][:recorder.last_iter_count]
    recorder.losses = _state_dict['losses'][:recorder.last_iter_count]
    recorder.values = _state_dict['values'][:start_epoch]
    recorder.iters = iters[:start_epoch]
  else:
    recorder.lrs = _state_dict['lrs']
    recorder.losses = _state_dict['losses']
    recorder.values = _state_dict['values']
    recorder.iters = iters
  if recorder.learn:
    recorder.learn.metrics = _state_dict['metrics']
  else:
    recorder.metrics = _state_dict['metrics']
  recorder.train_metrics = _state_dict['train_metrics']
  recorder.valid_metrics = _state_dict['valid_metrics']
  recorder.metric_names = _state_dict['metric_names']

@patch
def save(self:Recorder, file, metric_dir='metrics', ext='.meta'):
  if self.learn:
    file = join_path_file(file, Path(self.learn.path/self.learn.model_dir/metric_dir), ext)
  else:
    file = join_path_file(file, Path(metric_dir), ext)
  _save_recorder(file, self)
  
@patch
def load(self:Recorder, file, metric_dir='metrics', ext='.meta', start_epoch=None):
  if self.learn:
    file = join_path_file(file, Path(self.learn.path/self.learn.model_dir/metric_dir), ext)
  else:
    file = join_path_file(file, Path(metric_dir), ext)
  _restore_recorder(file, self, start_epoch)
  
@patch
def before_fit(self:Recorder):
    self.lrs = getattr(self, 'lrs', [])
    self.iters = getattr(self, 'iters', [])
    self.losses = getattr(self, 'losses', [])
    self.values = getattr(self, 'values', [])
    "Prepare state for training"
    if self.learn:
      names = self.learn.metrics.attrgot('name')
    else:
      names = self.metrics.attrgot('name')
    if self.train_metrics and self.valid_metrics:
        names = L('loss') + names
        names = names.map('train_{}') + names.map('valid_{}')
    elif self.valid_metrics: names = L('train_loss', 'valid_loss') + names
    else: names = L('train_loss') + names
    if self.add_time: names.append('time')
    self.metric_names = 'epoch'+names
    self.smooth_loss.reset()
    
from fastprogress.fastprogress import format_time
@patch
def after_epoch(self: Recorder):
  "Store and log the loss/metric values"
  if self.smooth_loss.count == 0 and len(self.values) > 0:
    self.log += self.values[self.epoch]
  self.learn.final_record = self.log[1:].copy()
  if self.add_time: self.log.append(format_time(time.time() - self.start_epoch))
  self.logger(self.log)
  if self.smooth_loss.count > 0:
    self.values.append(self.learn.final_record)
    self.iters.append(getattr(self, 'last_iter_count', 0) + self.smooth_loss.count) 
  
@patch
def plot_loss(self:Recorder, skip_start=5, with_valid=True, with_lr=True):
  fig, ax1 = plt.gcf(), plt.gca()
  ax1.plot(list(range(skip_start, len(self.losses))), self.losses[skip_start:], label='train')
  if with_valid:
    idx = (np.array(self.iters)<skip_start).sum()
    valid_col = self.metric_names.index('valid_loss') - 1
    ax1.plot(self.iters[idx:], L(self.values[idx:]).itemgot(valid_col), label='valid')
    ax1.scatter(self.iters[idx:], L(self.values[idx:]).itemgot(valid_col), c='red')
    ax1.legend(loc="best")
    ax1.set_ylabel('loss-value')
    ax1.set_xlabel('iter-loops')
  if with_lr:
    ax2 = ax1.twinx()
    ax2.plot(list(range(skip_start, len(self.lrs))), self.lrs[skip_start:], color='green', label='lr')
    ax2.set_ylabel('learning-rate')
    ax2.legend(loc="best")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    fig.subplots_adjust(right=0.85)


## save weights and meta-data after each loop
class AutoSaveCallback(Callback):
  order = 99
  def after_loss(self):
    if torch.isnan(self.learn.loss):
      self.learn.save(f'error')
      self.learn.recorder(f'error')
      getattr(self.learn, 'save_input')(f'error_input')
      exit()
    
  def after_epoch(self):
    if self.recorder.smooth_loss.count > 0:
      self.learn.save(f'{self.learn.epoch}')
      self.learn.recorder.save(f'{self.learn.epoch}')

## test-save-callback
class TestSaveCallback(Callback):
  order = 99
  saved = False
  def after_loss(self):
    if not self.saved:
      self.learn.save(f'error_{self.learn.epoch}')
      self.learn.recorder.save(f'error_{self.learn.epoch}')
      getattr(self.learn, 'save_input')(f'error_input_{self.learn.epoch}')
      self.saved = True

## plot train-loss while training
class AutoPlotCallback(Callback):
  order = 90
  def before_epoch(self):
    plt.ion()
  
  def after_batch(self):
    plt.clf()
    if self.learn.epoch > 900:
      epoch = self.learn.epoch
    self.learn.recorder.plot_loss()
    plt.pause(0.0001)

  def after_fit(self):
    plt.ioff()