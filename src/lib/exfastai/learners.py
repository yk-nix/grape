from fastai.data.all import *
from fastai.learner import *
from fastai.vision.all import *

__all__=['create_learner']

@patch
@delegates(Recorder.plot_loss)
def plot(self:Learner, epoch:int, with_lr=True, **kwargs):
  if self.recorder:
    self.recorder.load(str(epoch-1))
    self.recorder.plot_loss(with_lr=with_lr, **kwargs)

@patch
@delegates(load_model)
def fit(self:Learner, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0, device=None, **kwargs):
  if start_epoch != 0:
    self.load(str(start_epoch-1), device=device, **kwargs)
    recorder = getattr(self, 'recorder', None)
    if recorder:
      recorder.load(str(start_epoch-1))
    cbs = L(cbs) + SkipToEpoch(start_epoch)
  with self.added_cbs(cbs):
    if reset_opt or not self.opt: self.create_opt()
    if wd is None: wd = self.wd
    if wd is not None: self.opt.set_hypers(wd=wd)
    self.opt.set_hypers(lr=self.lr if lr is None else lr)
    self.n_epoch = n_epoch
    self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

@delegates(Learner.__init__)
def create_learner(name, dls, model, cbs, loss_func, opt_func, lr=1e-3, **kwargs):
  model_name = getattr(model, 'name', '')
  return Learner(dls, model, lr=lr, cbs=cbs, loss_func=loss_func, opt_func=opt_func, path=f'weights/{name}', model_dir=f'{model_name}', **kwargs)


