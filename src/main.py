from typing import NoReturn

from lib.learners.pedestrain import PedestrainDetector
from lib.learners.learner import Learner
from torch.optim.sgd import SGD

_models = {
  'PedestrainDecteor': PedestrainDetector
}


def reset_lr(detector: PedestrainDetector) -> NoReturn:
  learner = detector.learner
  learner.optimizer = SGD(learner.model.head.parameters(), lr = 0.0001, momentum = 0.9)
  learner.lr_scheduler = None

if __name__ == '__main__':
  detector = PedestrainDetector()
  detector.train('00061750.pth', load_cb = reset_lr)