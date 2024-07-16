from typing import Union, NoReturn
from pathlib import Path

import re

class Configurable:
  _CONF_DEFAULT = Path('cfg', 'grape.cfg')
  _STRIP_CHARS = ' \t\r\n"\''
  def __init__(self, config_file: Union[str, Path], separator: str = '='):
    self.config_file = Path(config_file) if config_file is not None else self._CONF_DEFAULT
    if not self.config_file.exists:
      raise RuntimeError(f'config file doesn\'t exist: {self.config_file}')
    self.config = {}
    self._separator = separator
    self.load()
  
  @property
  def separator(self) -> str:
    return self._separator
  
  @separator.setter
  def separator(self, separator: str) -> NoReturn:
    self._separator = separator

  def load(self) -> NoReturn:
    with open(self.config_file, 'r') as f:
      for line in f.readlines():
        line = re.compile('#.*').sub('', line).strip(self._STRIP_CHARS)
        if len(line) == 0:
          continue
        key, value = (e.strip(self._STRIP_CHARS) for e in line.split(self.separator))
        self.config[key] = value