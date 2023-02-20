import torch.nn as nn

from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

from torchvision.models.swin_transformer import SwinTransformer
from typing import Callable


def get_model(name) -> nn.Module:
	model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)

	model.eval()
	return model


def get_preprocess(name) -> Callable:
	return name.weigth.transforms()
