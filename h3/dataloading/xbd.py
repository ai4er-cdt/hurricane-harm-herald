import os
import shutil
from pathlib import Path
from glob import glob

from tqdm import tqdm

from h3 import logger
from h3.constants import SHA1_xbd
from h3.utils.directories import get_xbd_dir, get_xbd_disaster_dir
from h3.utils.file_ops import get_sha1, guarantee_existence, unpack_file
from h3.utils.simple_functions import convert_bytes


def sort_disaster_to_dir(disaster: str = "hurricane") -> None:
	"""
	Sort on disaster from the xBD_data into a folder of the name of the disaster.
	It keeps the same structure as the xBD dataset but with only one disaster.

	Parameters
	----------
	disaster : str, optional
		disaster to sort. The default is "hurricane"

	"""
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


def check_xbd(filepath: str, checksum: bool = False) -> bool:
	"""
	Check if a file exists, with the option to check its checksum.

	Parameters
	----------
	filepath : str
		The path of the file to check.
	checksum : bool, optional
		If True check the checksum of the file.
		The default is False.

	Notes
	-----
	The checksum only checks for the downloadable compressed parts and the combined compresse file.
	Will raise an error otherwise.

	Returns
	-------
	bool
		Returns False if the file does not exist.
		Returns True if the file does exist and if checksum is checked and matches.

	Raises
	------
	AssertionError
		If checksum is True and does not match.
	"""
	filename = os.path.basename(filepath)
	if not os.path.exists(filepath):
		logger.info(f"{filepath} does not exist.\nDownload them here: https://xview2.org/")
		return False
	if checksum:
		assert get_sha1(filepath) == SHA1[filename], f"{filename} failed sha1 checksum"
	return True


def _combine_xbd(
	output_filename: str = "xview2_geotiff.tgz",
	xbd_part_glob: str = "xview2_geotiff.tgz.part-*",
	delete_if_check: bool = False,
) -> None:
	"""
	Combine the different parts of xbd to one file (compressed).

	Parameters
	----------
	output_filename : str, optional
		The output of the combined file present in the xbd_dir. The file can already exist.
		The default is "xview2_geotiff.tgz".
	xbd_part_glob : str, optional
		glob name of the parts to merge present in the xbd_dir. Use `*` as a wild-card.
		The default is 'xview2_geotiff.tgz.part-*'.
	delete_if_check : bool, optional
		Option to delete the part file after combining. As a safety, it will only delete them if the checksum of
		the combined file matches the one expected. The default is False.
	"""
	xbd_dir = get_xbd_dir()
	output_filepath = os.path.join(xbd_dir, output_filename)

	# Note: sorted(Path(xbd_dir).glob(xbd_part_glob))
	# glob(xbd_part_glob, root_dir=xbd_dir)
	# but it x2 slower than the current method
	current_files = sorted(Path(xbd_dir).glob(xbd_part_glob))

	if not current_files:
		raise IOError("No files found\nMake sure you specified the correct glob name or have downloaded the files")

	if not os.path.exists(output_filepath):
		logger.info("Main xBD tar file not found, combining the part files.")
		with open(output_filepath, "wb") as wfb:
			for file in tqdm(current_files, desc="Combining xbd files"):
				filepath = os.path.join(xbd_dir, file)
				with open(filepath, "rb") as fd:
					shutil.copyfileobj(fd, wfb)  # similar to `cat FILE > NEW_FILE`
		logger.info(f"{len(current_files)} files merged into {output_filename}")

	if delete_if_check and check_xbd(output_filename, checksum=True):
		for file in tqdm(current_files, desc="Deleting"):
			os.remove(os.path.join(xbd_dir, file))
		logger.info("Combined file's checksum matches. Deleted the part files after merging.")


def _unpack_xbd(filename: str = "xview2_geotiff.tgz") -> None:
	# TODO: too slow
	xbd_dir = get_xbd_dir()
	filepath = os.path.join(xbd_dir, filename)
	unpack_file(filepath)


def get_xbd(
	checksum: bool = False,
	clean_after_merge: bool = False,
	unpack_tar: bool = True,
	combined_name: str = "xview2_geotiff.tgz",
) -> None:
	"""Wrapper function to check part files, combine and unpack them."""
	xbd_dir = get_xbd_dir()
	all_part_name = list(SHA1_xbd.keys())[1:]
	extracted_path = os.path.join(xbd_dir, "geotiffs")

	for name in all_part_name:
		filepath = os.path.join(xbd_dir, name)
		check_xbd(filepath=filepath, checksum=checksum)
	_combine_xbd(output_filename=combined_name, delete_if_check=clean_after_merge)
	if unpack_tar and not os.path.exists(extracted_path):
		_unpack_xbd(filename=combined_name)


def main():
	sort_disaster_to_dir()


if __name__ == "__main__":
	main()
