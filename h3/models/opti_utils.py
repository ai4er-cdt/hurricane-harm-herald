from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from numba import jit

from PIL import Image
from tqdm import tqdm
from h3.utils.directories import get_processed_data_dir

from typing import Callable


@jit(forceobj=True)
def open_img(path: str, transform: Callable):
	img = Image.open(path)
	img = transform(img)
	# print(img.shape)
	return img


def load_full_ram(lst_path: list, transform: Callable) -> list:
	all_object = []
	transforms = [transform for _ in range(len(lst_path))]
	with ThreadPoolExecutor() as pool:
		for results in list(tqdm(pool.map(open_img, lst_path, transforms), total=len(lst_path))):
			all_object.append(results)

	return all_object


def main():
	import glob
	import torch
	from torchvision.models import Swin_V2_B_Weights
	preprocessing = Swin_V2_B_Weights.IMAGENET1K_V1.transforms()

	zoom_levels = ["1", "2", "4", "0.5"]
	img_path = os.path.join(get_processed_data_dir(), "processed_xbd", "geotiffs_zoom", "images")
	img_paths = glob.glob(img_path + '/**/*.png', recursive=True)
	size = len(img_paths)
	all = load_full_ram(lst_path=img_paths[:int(size//96)], transform=preprocessing)
	print(len(all))


if __name__ == "__main__":
	main()
