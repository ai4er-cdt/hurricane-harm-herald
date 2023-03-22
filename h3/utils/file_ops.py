from __future__ import annotations

import hashlib
import json
import os
import shutil

from h3 import logger


def guarantee_existence(path: str) -> str:
	"""Function to guarantee the existence of a path, and returns its absolute path.

	Parameters
	----------
	path : str
		Path (in str) to guarantee the existence.

	Returns
	-------
	str
		The absolute path.
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	return os.path.abspath(path)


def get_sha1(filepath: str) -> str:
	"""
	As the files are big using this method that uses buffers.

	Parameters
	----------
	filepath : str
		Filepath of the file to calculate the SHA1.

	Returns
	-------
	str
		The SHA1 of the file.

	References
	----------
	https://stackoverflow.com/a/22058673/9931399
	"""
	BUF_SIZE = 65536    # chunks of 64kb
	sha1 = hashlib.sha1()
	with open(filepath, "rb") as f:
		while True:
			data = f.read(BUF_SIZE)
			if not data:
				break
			sha1.update(data)
	return sha1.hexdigest()


def unpack_file(filepath: str, clean: bool = False, file_format: None | str = None) -> None:
	"""Unpack an archive file.
	It is quite slow for big files

	Parameters
	----------
	filepath : str,
		Path of the file to unpack, it will unpack in the folder
	clean : bool, optional
		If True will delete the archive after unpacking. The default is False.
	file_format : str, optional
		The archive format. If None it will use the file extension.
		See shutil.unpack_archive()

	Notes
	-----
	It is quite slow for big files.
	"""
	# TODO: this is a bit slow, and not verbose
	logger.info(f"Unpacking {os.path.basename(filepath)}\nThis can take some time")
	shutil.unpack_archive(filepath, extract_dir=os.path.dirname(filepath), format=file_format)
	logger.info(f"{os.path.basename(filepath)} unpack in {os.path.dirname(filepath)}")
	if clean:
		logger.debug(f"Deleting {os.path.basename(filepath)}")
		os.remove(filepath)


def model_run_to_json(start_time: str | None, end_time: str | None, run_parameters) -> None:
	from h3.utils.directories import get_datasets_dir
	model_json_file = os.path.join(get_datasets_dir(), "model_runs.json")

	new_id = 0

	if os.path.exists(model_json_file):
		with open(model_json_file, "r") as f:
			models_json = json.load(f)
		new_id = str(max(map(int, list(models_json.keys()))) + 1)

	new_run_json = {
		"checkpoint_name": run_parameters["ckp_name"],
		"start_date": start_time,
		"end_date": end_time,
		"split_val_train_test": run_parameters["split_val_train_test"],
		"EF_features": {
			"nbr": sum(map(len, run_parameters["ef_features"].values())),
			"values": run_parameters["ef_features"]
		},
		"architecture": run_parameters["image_embedding_architecture"],
		"zoom_levels": run_parameters["zoom_levels"],
		"is_balanced": run_parameters["balanced"],
		"max_epochs": run_parameters["max_epochs"],
		"is_augmented": bool(run_parameters["use_augmentation"]) if run_parameters["use_augmentation"] is not None else None,
		"ram_load": run_parameters["ram_load"],
		"is_spatial": run_parameters["spatial"],
		"hurricanes": run_parameters["hurricanes"],
		"precision": run_parameters["precision"],
		"torch_float32_precision": run_parameters["torch_float32_precision"],
		"image_only": run_parameters["image_only_model"],
		"loss_function": run_parameters["loss_function_str"],
		"from_checkpoint": run_parameters["from_checkpoint"]
	}

	logger.info("Writing parameters to json file")
	with open(model_json_file, "w") as f:
		models_json[new_id] = new_run_json
		json.dump(models_json, f, indent=2)


def get_non_empty_subfolder(path: str) -> str | None:
	"""Returns the name of the non-empty subfolder within the given path.

	Assuming one and only one of the subfolders is non-empty.

	Parameters
	----------
	path: str
		path of the directory containing the subfolders.

	Returns
	-------
	str or None
		The name of the non-empty subfolder, or None if no non-empty subfolder is found.
	"""
	for folder in os.listdir(path):
		if len(os.listdir(os.path.join(path, folder))):
			return folder


def main():
	pass


if __name__ == "__main__":
	main()
