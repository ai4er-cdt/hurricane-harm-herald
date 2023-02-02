import os
import shutil
import tarfile
from pathlib import Path
from glob import glob

from tqdm import tqdm
from h3.utils.directories import get_xbd_dir, get_xbd_disaster_dir
from h3.utils.file_ops import get_sha1, guarantee_existence
from h3.utils.simple_functions import convert_bytes

SHA1 = {
	"xview2_geotiff.tgz": "6eae3baddf86796c15638682a6432e3e6223cb39",
	"xview2_geotiff.tgz.part-aa": "881ed94d1060c91e64c8eae438dfce492a21d9a9",
	"xview2_geotiff.tgz.part-ab": "4064dddc9aa05f786a3a6f70dd4ca86d79dd9e3a",
	"xview2_geotiff.tgz.part-ac": "0cfdf761e6f77ac5c423d9fb0927c3f8f8ac43da",
	"xview2_geotiff.tgz.part-ad": "44a39a7c4a80d386fb71ced95caee040126bb405",
	"xview2_geotiff.tgz.part-ae": "7fb96fac1d009b6a213d4efef2fbf5f1a475a554",
	"xview2_geotiff.tgz.part-af": "2ccbd04c4b2e27f8d948de734f661ec0c9d81152"
}


def sort_disaster_to_dir(disaster: str = "hurricane") -> None:
	xbd_dir = get_xbd_dir()
	disaster_dir = get_xbd_disaster_dir(disaster)
	all_disaster = glob(f"**/{disaster}-*", root_dir=xbd_dir, recursive=True)
	total_size = 0

	for file in tqdm(all_disaster, desc=f"Sorting files of {disaster}", unit="file"):
		abs_filepath = os.path.join(xbd_dir, file)
		path = Path(file)
		path_part = path.parts

		# change the folder to "disaster" instead of the default "geotiffs"
		new_path = os.path.join(disaster_dir, *path_part[1:-1])
		file_name = path_part[-1]

		guarantee_existence(new_path)
		new_filepath = os.path.join(new_path, file_name)
		shutil.copyfile(abs_filepath, new_filepath)

		total_size += os.path.getsize(abs_filepath)

	print(f"Copied {convert_bytes(total_size)}")


def check_xbd(filename: str, checksum: bool = False) -> bool:
	filepath = filename
	if not os.path.exists(filepath):
		raise IOError(f"{filename} does not exist")
	if checksum:
		assert get_sha1(filepath) == SHA1[filename], f"{filename} failed sha1 checksum"
	return True


def combine_xbd(
		output_filename: str = "xview2_geotiff.tgz",
		xbd_part_glob: str = "xview2_geotiff.tgz.part-*",
		delete_if_check: bool = False
) -> None:
	"""

	Parameters
	----------
	output_filename : str, optional
	xbd_part_glob : str, optional
	delete_if_check : bool, optional
		Option to delete the part file after combining. As a safety, it will only delete them if the checksum of
		the combined file matches the one expected. The default is False.

	"""
	xbd_dir = get_xbd_dir()
	output_filepath = os.path.join(xbd_dir, output_filename)

	current_files = glob(xbd_part_glob, root_dir=xbd_dir)

	if not current_files:
		raise IOError("No files found\nMake sure you specified the correct glob name or have downloaded the files")

	with open(output_filepath, "wb") as wfb:
		for file in tqdm(current_files, desc="File"):
			filepath = os.path.join(xbd_dir, file)
			with open(filepath, "rb") as fd:
				shutil.copyfileobj(fd, wfb)     # similar to `cat FILE > NEW_FILE`

	if delete_if_check and check_xbd(output_filename, checksum=True):
		for file in tqdm(current_files, desc="Deleting"):
			os.remove(os.path.join(xbd_dir, file))


def unpack_xbd(filename: str = "xview2_geotiff.tgz") -> None:
	# TODO: too slow
	xbd_dir = get_xbd_dir()
	filepath = os.path.join(xbd_dir, filename)

	print(f"Unpacking {filename}\nThis can take some time")
	with tarfile.open(filepath, "r:gz") as tar:
		tar.extractall(path=xbd_dir)


def get_xbd():
	pass


def main():
	sort_disaster_to_dir()


if __name__ == '__main__':
	main()
