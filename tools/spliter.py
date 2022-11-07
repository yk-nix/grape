import os
import random
import time
import os.path as path
import argparse

def split(train_pct:float, valid_pct:float, inp_dir, out_dir, seed=None):
  names = ['train.txt', 'val.txt', 'test.txt']
  items = os.listdir(inp_dir)
  size = len(items)
  if seed is None:
    seed = int(time.time())
  stats = random.getstate()
  idxs = random.sample(range(size), k=size)
  random.setstate(stats)
  cut1 = int(size * train_pct)
  cut2 = int(size * valid_pct)
  for name, idx in zip(names, (idxs[:cut1], idxs[cut1:cut2+cut1], idxs[cut2+cut1:])):
    with open(path.join(out_dir, name), 'w') as file:
      file.writelines([items[i]+'\n' for i in idx])
      #file.write("\n".join([items[i] for i in idx]))
  print(f'split success. total={size}, tarin={cut1}, valid={cut2}, test={size-cut1-cut2}')
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='split',
                                   description='split data into train/valid/test')
  parser.add_argument('-o', '--out_dir', help='directory where the output files will be located.', default='.')
  parser.add_argument('--valid_pct', type=float, help='how much the valid data should be.', default=0.)
  parser.add_argument('--test_pct', type=float, help='how much the test data should be.', default=0.)
  parser.add_argument('--seed', type=int, help='seed would be set to random stat.e')
  parser.add_argument('inp_dir', help='the directory in which the datum are located in.')
  args = parser.parse_args()
  split(train_pct = 1.0 - args.valid_pct - args.test_pct,
        valid_pct = args.valid_pct,
        inp_dir = args.inp_dir,
        out_dir = args.out_dir,
        seed = args.seed)