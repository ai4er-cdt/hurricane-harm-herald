import os.path

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from h3.utils.directories import get_xbd_hurricane_dir


def load_image() -> np.ndarray:
	hurricane_dir = get_xbd_hurricane_dir()
	imagepath = os.path.join(hurricane_dir, "hold", "images", "hurricane-florence_00000236_post_disaster.tif")
	src = rasterio.open(imagepath)
	image = src.read()
	return image


def load_model():
	# model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
	model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
	print(type(model))

	model.eval()
	return model


def main():
	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print("loading model")
	model = load_model()
	print("loading image")
	image = load_image()
	print(image.shape)

	image = torch.as_tensor(image)

	# preprocess = ViT_L_16_Weights.IMAGENET1K_V1.transforms()
	preprocess = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()

	batch = preprocess(image).unsqueeze(0)

	out = model(batch).squeeze(0)
	print(out)
	print(out.shape)


if __name__ == "__main__":
	main()
