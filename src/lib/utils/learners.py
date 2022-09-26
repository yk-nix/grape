from fastai.data.all import *
from fastai.learner import *
from fastai.vision.all import *

__all__=['create_learner']

@delegates(Learner.__init__)
def create_learner(dls, model, cbs, name, lr=1e-3, start_epoch=None, **kwargs):
  model_name = getattr(model, 'name', '')
  return Learner(dls, model, lr=lr, cbs=cbs, loss_func=CrossEntropyLossFlat(), path=f'models/{name}', model_dir=f'{model_name}',
                 opt_func=partial(SGD, mom=0.9), **kwargs)