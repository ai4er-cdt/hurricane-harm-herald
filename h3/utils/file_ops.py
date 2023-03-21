from __future__ import annotations

import hashlib
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


def main():
	pass


if __name__ == "__main__":
	main()


