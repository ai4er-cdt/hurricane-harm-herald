import os
from glob import glob

from tqdm import tqdm
from h3.utils.directories import get_xbd_dir
from h3.utils.file_ops import get_sha1

SHA1 = {
	"xview2_geotiff.tgz": "6eae3baddf86796c15638682a6432e3e6223cb39",
	"xview2_geotiff.tgz.part-aa": "881ed94d1060c91e64c8eae438dfce492a21d9a9",
	"xview2_geotiff.tgz.part-ab": "4064dddc9aa05f786a3a6f70dd4ca86d79dd9e3a",
	"xview2_geotiff.tgz.part-ac": "0cfdf761e6f77ac5c423d9fb0927c3f8f8ac43da",
	"xview2_geotiff.tgz.part-ad": "44a39a7c4a80d386fb71ced95caee040126bb405",
	"xview2_geotiff.tgz.part-ae": "7fb96fac1d009b6a213d4efef2fbf5f1a475a554",
	"xview2_geotiff.tgz.part-af": "2ccbd04c4b2e27f8d948de734f661ec0c9d81152"
}


def pick_disaster_to_dir(disaster: str = "hurricane") -> None:
	xbd_dir = get_xbd_dir()


def check_xbd(filename: str, checksum: bool = False) -> None:
	filepath = filename
	if checksum:
		assert get_sha1(filepath) == SHA1[filename], f"{filename} failed sha1 checksum"


def get_xbd():
	pass


def unpack_xbd():
	pass
