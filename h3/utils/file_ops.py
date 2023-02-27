import hashlib
import os
import tarfile
import zipfile

from h3 import logger


def guarantee_existence(path: str) -> str:
	if not os.path.exists(path):
		os.makedirs(path)
	return os.path.abspath(path)


def get_sha1(filepath: str) -> str:
	"""
	As the files are big using this method that uses buffers
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


def unpack_file(filepath: str, mode: str):
	"""
	Unpack a tar file.
	It is quite slow for big files

	Parameters
	----------
	filepath : str,
		Path of the file to unpack, it will unpack in the folder
	mode : str,
		mode to open the zip/tar file with
	"""
	# TODO: this is a bit slow, and not verbose
	logger.info(f"Unpacking {os.path.basename(filepath)}\nThis can take some time")
	with tarfile.open(filepath, mode) as tar:
		tar.extractall(path=os.path.dirname(filepath))
