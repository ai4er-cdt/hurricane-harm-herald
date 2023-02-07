import hashlib
import os


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
