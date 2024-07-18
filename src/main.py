from typing import NoReturn

from lib.learners.pedestrain import PedestrainDetector
from lib.learners.learner import Learner

from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import MultiStepLR


_models = {
  'PedestrainDecteor': PedestrainDetector
}


def reset_otpimizer_and_lr_scheduler(detector: PedestrainDetector) -> NoReturn:
  learner = detector.learner
  learner.optimizer = SGD(learner.model.head.parameters(), lr = 0.01, momentum = 0.9)
  learner.lr_scheduler = MultiStepLR(learner.optimizer, [100, 500, 1000, 2000])

def reset_lr_scheduler(detector: PedestrainDetector) -> NoReturn:
  learner = detector.learner
  learner.lr_scheduler = MultiStepLR(learner.optimizer, [])

if __name__ == '__main__':
  detector = PedestrainDetector()
  detector.train('00230744.pth', after_load_cb = reset_otpimizer_and_lr_scheduler)
  # detector.test('00228796.pth', image_set = 'val')