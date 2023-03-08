from __future__ import annotations

import os
import datetime
import xarray as xr

from pathlib import Path

from h3 import logger
from h3.utils.directories import get_cmorph_dir, get_h3_dir
from h3.utils.downloader import downloader


def _download_cmorph() -> None:
	logger.debug("Downloading the cmorph files")
	urls_file = os.path.join(get_h3_dir(), "h3", "_config", "CMORPH_urls.txt")
	# Nice way to read multiple lines without the \n
	urls_to_dl = Path(urls_file).read_text().splitlines()
	downloader(urls_to_dl, get_cmorph_dir())
	logger.debug("File saved.")


def cmorph_filename_range(start_date: datetime.date, stop_date: datetime.date) -> list:
	pass


def load_cmorph(hurricane_date: datetime.date, day_range: int):
	cmorph_dir = get_cmorph_dir()
	if os.listdir(cmorph_dir) == 0:
		logger.debug("The cmorph files are not present.")
		_download_cmorph()

	start_date = hurricane_date - datetime.timedelta(day_range)
	stop_date = hurricane_date + datetime.timedelta(day_range)

	file_to_load = cmorph_filename_range(start_date, stop_date)
	weather_data = xr.open_mfdataset(file_to_load, parallel=True)


def main():
	_download_cmorph()


if __name__ == "__main__":
	main()
