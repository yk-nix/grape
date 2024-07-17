from typing import NoReturn

from lib.learners.pedestrain import PedestrainDetector
from lib.learners.learner import Learner

from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import MultiStepLR


_models = {
  'PedestrainDecteor': PedestrainDetector
}


def reset_lr(detector: PedestrainDetector) -> NoReturn:
  learner = detector.learner
  learner.optimizer = SGD(learner.model.head.parameters(), lr = 0.01, momentum = 0.9)
  learner.lr_scheduler = MultiStepLR(learner.optimizer, [100, 500, 1000, 2000])

if __name__ == '__main__':
  detector = PedestrainDetector()
  detector.train('00154146.pth', load_cb = reset_lr, ignore_lr_scheduler = True)
  # detector.test('00147528.pth', image_set = 'train')