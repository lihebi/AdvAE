# tqdm has too many bugs
from tqdm import tqdm

__all__ = ['clear_tqdm']

def clear_tqdm():
    if '_instances' in dir(tqdm):
        tqdm._instances.clear()

