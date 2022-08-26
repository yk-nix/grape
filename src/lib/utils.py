from fastai.data.all import *
from fastai.vision.all import *
from xml.dom.minidom import parse

def get_annotations_voc(path: Any):
  xmls = get_files(path, extensions='.xml', folders=['annotations', 'Annotations'])
  images, bbox_labels = [], []
  for xml in xmls:
    with open(xml) as xml_file:
      doc = parse(xml_file).documentElement
    images.append(doc.getElementsByTagName('filename')[0].childNodes[0].data)
    objs = doc.getElementsByTagName('object')
    bboxes, lbls = [], []
    for obj in objs:
      lbls.append(obj.getElementsByTagName('name')[0].childNodes[0].data)
      xmin = obj.getElementsByTagName('xmin')[0].childNodes[0].data
      ymin = obj.getElementsByTagName('ymin')[0].childNodes[0].data
      xmax = obj.getElementsByTagName('xmax')[0].childNodes[0].data
      ymax = obj.getElementsByTagName('ymax')[0].childNodes[0].data
      bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
    bbox_labels.append((bboxes, lbls))
  return images, bbox_labels

def get_metric_file_name(wd:Any, recorder_dir, model_name, start_epoch:int, end_epoch:int):
  return os.path.join(wd, recorder_dir, f'{model_name}_{start_epoch:03}-{end_epoch+1:03}.meta')

def save_metrics(recorder: Recorder, model_name:str, recorder_dir='metrics'):
  data = {
    'lrs': recorder.lrs,
    'iters': recorder.iters,
    'losses': recorder.losses,
    'loss': recorder.loss,
    'values': recorder.values,
    'smooth_loss': recorder.smooth_loss,
    'add_time': recorder.add_time,
    'metrics': recorder.metrics,
    'train_metrics': recorder.train_metrics,
    'valid_metrics': recorder.valid_metrics,
    'metric_names': recorder.metric_names,
    'log': recorder.log
  }
  torch.save(data, get_metric_file_name(recorder.learn.path, recorder_dir, model_name, recorder.learn.start_epoch, recorder.learn.epoch))


def load_metrics(learn: Learner, model_name:str, start_epoch:int,  end_epoch:int, recorder_dir='metrics'): 
  recorder = getattr(learn, 'recorder', None)
  if recorder and isinstance(recorder, Recorder):
    data = torch.load(get_metric_file_name(learn.path, recorder_dir, model_name, start_epoch, end_epoch - 1))
    recorder.lrs = data['lrs']
    recorder.iters = data['iters']
    recorder.losses = data['losses']
    recorder.loss = data['loss']
    recorder.values = data['values']
    recorder.smooth_loss = data['smooth_loss']
    recorder.add_time = data['add_time']
    recorder.metrics = data['metrics']
    recorder.train_metrics = data['train_metrics']
    recorder.valid_metrics = data['valid_metrics']
    recorder.metric_names = data['metric_names']
    recorder.log = data['log']    


def plot_metrics(learn: Learner, model_name:str, start_epoch:int, end_epoch:int, recorder_dir='metrics'):
  metric_dir = os.path.join(learn.path, recorder_dir)
  metric_files = L(os.listdir(metric_dir)).filter(lambda fn: fn.startswith(model_name)).sorted()
  iters, losses, values = [], [], []
  if len(metric_files) <= 0:
    raise ValueError(f'not found any metric files for model: {model_name}')
  count, last_epoch = 0, 0
  for metric_file in metric_files:
    data = torch.load(os.path.join(metric_dir, metric_file))
    search_ret = re.search(r'(.*)_(\d+)-(\d+).meta', metric_file)
    if search_ret:
      _start = int(search_ret.group(2))
      _end = int(search_ret.group(3))
      if start_epoch >= _end:
        continue
      if start_epoch > _start:
        start_epoch = _start
      losses.extend(data['losses'])
      iters.extend(L(data['iters']).filter(lambda e: e > 0).map(lambda e: e + count))
      values.extend(L(data['values']).filter(lambda e: len(e) > 0))
      count = count + data['iters'][-1]
      if last_epoch < _end:
        last_epoch = _end
      if end_epoch <= _end:
        end_epoch = _end
        break
  if last_epoch < end_epoch:
    end_epoch = last_epoch
  plt.plot(list(range(len(losses))), losses, label='train')
  plt.plot(iters, L(values).itemgot(1), label='valid')
  plt.legend()
  plt.title(f'epoch: {start_epoch}-{end_epoch}')
  # d = torch.load(os.path.join(metric_dir, metric_files[0]))
  # n_loops = d['iters'][0]
  # for metric_file in metric_files:
  #   search_ret = re.search(r'(.*)_(\d+)-(\d+).meta', metric_file)
  #   if search_ret:
  #     _start = int(search_ret.group(2))
  #     _end = int(search_ret.group(3))
  #     if start_epoch >= _end:
  #       continue
  #     data = torch.load(os.path.join(metric_dir, metric_file))
  #     for iter in data['iters']:
  #       count = count + 1
  #       if iter == 0:
  #         iters.append(0)
  #       else:
  #         iters.append(n_loops * count)
  #     losses.extend(data['losses'])
  #     values.extend(data['values'])
  #     if end_epoch <= _end:
  #       break    
          
          
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
  def __init__(self, model_name:str):
    self.model_name = model_name.strip() if model_name is not None else ''
    
  def after_epoch(self):
    if self.model_name is None or len(self.model_name) == 0:
      self.learn.save(f'{self.learn.epoch + 1:03}')
    else:
      self.learn.save(f'{self.model_name}/{self.learn.epoch + 1:03}')
  
  def after_fit(self):
    save_metrics(self.learn.recorder, self.model_name)