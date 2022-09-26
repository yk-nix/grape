from fastai.data.all import *
from fastai.learner import *





# @delegates(Learner.__init__)
# def mnist_classifier(data=None, model=None, lr=1e-3, model_dir='models/cls_mnist', start_epoch=None, **kwargs):
#   if data is None: 
#     data = get_dls_mnist()
#   if model is None:
#     model = LeNet()
#   learn = Learner(data, model, lr=lr, loss_func=CrossEntropyLossFlat(),
#                  opt_func=partial(SGD, mom=0.9), **kwargs)
  
#   if start_epoch and start_epoch > 0:
#     weight_file_name = f'{model_name}/{start_epoch:03}'
#     learn.load(weight_file_name)
#     learn.start_epoch = start_epoch
#   else:
#     learn.start_epoch = 0
#   learn.add_cbs([AutoSaveCallback(model_name='od_voc')])
#   return learn