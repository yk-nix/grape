import argparse

from lib.models.all import *
from lib.exfastai.all import *

__d = {
  'lenet': ('mnist', LeNet(num_class=10))
}

def save_as_darknet(weight_file, model, major, minor, revision, iters):
  f = getattr(model, 'to_darknet', None)
  if f is None:
    print(f'{model.name} has not implemented to_darknet.')
    return
  f(weight_file, major, minor, revision, iters)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='todarkent', description='convert weight file into darknet-format.')
  parser.add_argument('--major', type=int, default=0, help='major version, default is 0.')
  parser.add_argument('--minor', type=int, default=0, help='minor version, default is 2')
  parser.add_argument('--revision', type=int, default=0, help='revision number, default is 0.')
  parser.add_argument('--iters', type=int, default=-1, help='how many iters has been trained for this model.')
  parser.add_argument('model', help='what model you want to convert.')
  parser.add_argument('epoch', type=int, help='which epoch weight file you want to convert.')
  parser.add_argument('weight_file', type=str, help='where to save the converted file.')
  args = parser.parse_args()
  if args.model not in __d.keys():
    print(f'{args.model} is not supported. only the folloing models are supported:')
    print(f'\t{list(__d.keys())}')
  name, model = __d[args.model]
  learn = create_learner(name, None, model, [], object(), object())
  learn.load(f'{args.epoch:03d}', with_opt=False)
  save_as_darknet(args.weight_file, model, args.major, args.minor, args.revision, args.iters)