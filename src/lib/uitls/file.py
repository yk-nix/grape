from typing import Union
from pathlib import Path

__all__ = ['join_path_file']


def join_path_file(*path: Union[str, Path], file: Union[str, Path], ext: str = ''):
  '''
    Return `path/file` if file is a string or a `Path`, file otherwise;
    and create parent-directoris if neccessary. 
  '''
  if not isinstance(file, (str, Path)): return file
  directory = Path(*path)
  directory.mkdir(parents = True, exist_ok = True)
  return Path(directory, f'{file}{ext}')