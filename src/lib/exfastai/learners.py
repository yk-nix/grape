from fastai.data.all import *
from fastai.learner import *
from fastai.vision.all import *

__all__=['create_learner']

#-------------------------------------------------------------------------
# override Learner
# Cell
@patch
@delegates(load_model)
def fit(self:Learner, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0, device=None, **kwargs):
  self.load(f'{start_epoch:03d}', device=device, **kwargs)
  if start_epoch != 0:
    recorder = getattr(self, 'recorder', None)
    if recorder is not None:
      recorder.load(f'{start_epoch:03d}')
    cbs = L(cbs) + SkipToEpoch(start_epoch)
  with self.added_cbs(cbs):
    if reset_opt or not self.opt: self.create_opt()
    if wd is None: wd = self.wd
    if wd is not None: self.opt.set_hypers(wd=wd)
    self.opt.set_hypers(lr=self.lr if lr is None else lr)
    self.n_epoch = n_epoch
    self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

@patch
def _get_file_path(self:Learner, file_name, sub_dir, ext):
  if os.path.isabs(file_name):
    return Path(file_name)
  if sub_dir:
    return join_path_file(file_name, Path(self.path/self.model_dir/sub_dir), ext)
  else:
    return join_path_file(file_name, Path(self.path/self.model_dir), ext)

@patch
def save_input(self:Learner, file_name, sub_dir=None, ext=".pth"):
  input = {
    'xb': getattr(self, 'xb', None),
    'yb': getattr(self, 'yb', None)
  }
  file = self._get_file_path(file_name, sub_dir, ext)
  torch.save(input, file)

@patch
def load_input(self:Learner, file_name, sub_dir=None, ext=".pth"):    
  file = self._get_file_path(file_name, sub_dir, ext)
  input = torch.load(file)
  if 'xb' in input.keys():
    setattr(self, 'xb', input['xb'])
  if 'yb' in input.keys():
    setattr(self, 'yb', input['yb'])

@patch
@delegates(save_model)
def save(self:Learner, file_name, sub_dir=None, ext='.pth', **kwargs):
    "Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`"
    file = self._get_file_path(file_name, sub_dir, ext)
    save_model(file, self.model, getattr(self,'opt',None), **kwargs)
    return file

@patch
@delegates(load_model)
def load(self:Learner, file_name, sub_dir=None, ext='.pth', with_opt=True, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`"
    if device is None and hasattr(self.dls, 'device'): device = self.dls.device
    if self.opt is None and with_opt: self.create_opt()
    file = self._get_file_path(file_name, sub_dir, ext)
    load_model(file, self.model, self.opt, device=device, with_opt=with_opt, **kwargs)
    nested_attr(self, "accelerator.wait_for_everyone", noop)()
    return self
  
  
#-------------------------------------------------------------------------
# learn creator
@delegates(Learner.__init__)
def create_learner(name, dls, model, cbs, loss_func, opt_func, lr=1e-3, **kwargs):
  model_name = getattr(model, 'name', '')
  return Learner(dls, model, lr=lr, cbs=cbs, loss_func=loss_func, opt_func=opt_func, path=f'weights/{name}', model_dir=f'{model_name}', **kwargs)


