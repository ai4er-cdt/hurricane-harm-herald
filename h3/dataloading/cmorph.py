import os

from pathlib import Path

from h3 import logger
from h3.utils.directories import get_cmorph_dir, get_h3_dir
from h3.utils.downloader import downloader


def _download_cmorph():
	logger.debug("Downloading the cmorph files")
	urls_file = os.path.join(get_h3_dir(), "h3", "_config", "CMORPH_urls.txt")
	# Nice way to read multiple lines without the \n
	urls_to_dl = Path(urls_file).read_text().splitlines()

	downloader(urls_to_dl, get_cmorph_dir())


def load_cmorph():
	cmorph_dir = get_cmorph_dir()
	if os.listdir(cmorph_dir) == 0:
		logger.debug("The cmorph files are not present.")
	pass


def main():
	_download_cmorph()


if __name__ == "__main__":
	main()
