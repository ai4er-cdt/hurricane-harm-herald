import torch
from pytorch_lightning import Trainer, seed_everything


from ._config import *

seed_everything(17, workers=True)