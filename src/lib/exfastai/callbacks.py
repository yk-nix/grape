from posixpath import isabs
from fastai.data.all import *
from fastai.callback.all import *
from fastai.learner import *

from fastprogress.fastprogress import format_time

__all__=['AutoPlotCallback']

#----------------------------------------------------------------------------
# override Recorder
## plot train-loss while training
class PlotLossCallback(Callback):
  order = 90
  def before_epoch(self):
    plt.ion()
  
  def after_batch(self):
    plt.clf()
    self.learn.recorder.plot_loss()
    plt.pause(0.001)

  def after_fit(self):
    plt.ioff()



#----------------------------------------------------------------------------
# override Recorder
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
def __init__(self:Recorder, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98, auto_save=True, auto_save_error=False):
  store_attr('add_time,train_metrics,valid_metrics,auto_save,auto_save_error')
  self.loss,self.smooth_loss = AvgLoss(),AvgSmoothLoss(beta=beta)
  
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
    
@patch
def before_epoch(self:Recorder):
  "Set timer if `self.add_time=True`"
  self.cancel_train,self.cancel_valid = False,False
  if self.add_time: self.start_epoch = time.time()
  self.log = L(getattr(self, 'epoch', 0)+1)
  
@patch
def after_loss(self:Recorder):
  if torch.isnan(self.learn.loss) or torch.isinf(self.learn.loss):
    self.batch_canceled = True
    if self.auto_save_error:
      self.save_error()
    raise CancelBatchException
    
@patch
def after_batch(self:Recorder):
  batch_canceled = getattr(self, 'batch_canceled', False)
  if batch_canceled:  # if this batch is canceled, do not recorde its loss value
    self.batch_canceled = False
    return
  "Update all metrics and records lr and smooth loss in training"
  if len(self.yb) == 0: return
  mets = self._train_mets if self.training else self._valid_mets
  for met in mets: met.accumulate(self.learn)
  if not self.training: return
  self.lrs.append(self.opt.hypers[-1]['lr'])
  self.losses.append(self.smooth_loss.value)
  self.learn.smooth_loss = self.smooth_loss.value
  
@patch
def _get_file_path(self:Recorder, file_name, sub_dir, ext):
  if os.path.isabs(file_name):
    return file_name
  if self.learn:
    file = join_path_file(file_name, Path(self.path/self.model_dir/sub_dir), ext)
  else:
    file = join_path_file(file_name, Path(sub_dir), ext)
  return file

@patch
def save_error(self:Recorder, model_file_name=None, input_file_name=None,  sub_dir='errors', ext='.pth', save_model=True, save_input=True):
  if self.learn:
    suffix = f'{self.learn.epoch+1:03d}_{self.learn.iter:05d}'
    if save_model: 
      model_file_name = f'weight_{suffix}' if model_file_name is None else model_file_name
      self.learn.save(file_name=model_file_name, sub_dir=sub_dir, ext=ext)
    if save_input:
      input_file_name = f'input_{suffix}' if input_file_name is None else input_file_name
      self.learn.save_input(file_name=input_file_name, sub_dir=sub_dir, ext=ext)
    
@patch
def load_error(self:Recorder, model_file_name=None, input_file_name=None, sub_dir='erros', ext='.pth', load_model=True, load_input=True):
  if self.learn:
    if load_model: self.learn.load(file_name=model_file_name, sub_dir=sub_dir, ext=ext)
    if load_input: self.learn.load_input(file_name=input_file_name, sub_dir=sub_dir, ext=ext)

@patch
def save(self:Recorder, file_name, metric_dir='metrics', ext='.meta'):
  file = self._get_file_path(file_name, sub_dir=metric_dir, ext=ext)
  _save_recorder(file, self)
  
@patch
def load(self:Recorder, file_name, metric_dir='metrics', ext='.meta', start_epoch=None):
  file = self._get_file_path(file_name, sub_dir=metric_dir, ext=ext)
  _restore_recorder(file, self, start_epoch)

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
  if self.auto_save:
    suffix = f'{self.learn.epoch+1:03d}'
    self.learn.save(suffix)
    self.learn.recorder.save(suffix)
  
@patch
def plot_loss(self:Recorder, skip_start=5, with_valid=True, with_lr=False):
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


#---------------------------------------------------------------------------------------------
## override callback: ProgressCallback.after_batch
@patch
def after_batch(self:ProgressCallback):
  self.pbar.update(self.iter+1)
  if hasattr(self, 'smooth_loss'): self.pbar.comment = f'{self.loss:.4f}->{self.smooth_loss:.4f}'