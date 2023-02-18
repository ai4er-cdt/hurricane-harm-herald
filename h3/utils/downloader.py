from __future__ import annotations

import os.path
import urllib.request
import urllib.error
# import signal

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPResponse
# from threading import Event
from tqdm.auto import tqdm

from typing import Iterable, Generator

from h3 import logger
from h3.utils.directories import get_download_dir


CHUNK_SIZE = 1024
# done_event = Event()
#
#
# def handle_signals(signum, handle):
# 	done_event.set()
#
#
# signal.signal(signal.SIGINT, handle_signals)


def _get_response_size(resp: HTTPResponse) -> None | int:
	"""
	Get the size of the file to download
	"""
	try:
		return int(resp.info()["Content-length"])
	except (ValueError, KeyError, TypeError):
		return None


def _get_chunks(resp: HTTPResponse) -> Generator[bytes, None]:
	"""
	Generator of the chunks to download
	"""
	while True:
		chunk = resp.read(CHUNK_SIZE)
		if not chunk:
			break
		yield chunk


def url_download(url: str, path: str) -> None:
	"""
	Download an url to a local file
	"""
	response = urllib.request.urlopen(url)
	chunks = _get_chunks(response)
	pbar = tqdm(
		desc=f"Requesting {os.path.basename(url)}",
		unit="B",
		total=_get_response_size(response),
		unit_scale=True,
		# format to have current/total size with the full unit, e.g. 60kB/6MB
		# https://github.com/tqdm/tqdm/issues/952
		bar_format="{l_bar}{bar}| {n_fmt}{unit}/{total_fmt}{unit}"
		" [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
	)
	with pbar as t:
		with open(path, "wb") as file:
			for chunk in chunks:
				file.write(chunk)
				t.update(len(chunk))
				# if done_event.is_set():
				# 	return
	logger.info(f"Downloaded in {path}")


def downloader(urls: Iterable[str], target_dir: str = get_download_dir()):
	"""
	Downloader to download multiple files
	"""
	with ThreadPoolExecutor(max_workers=4) as pool:
		target_dir = os.path.abspath(target_dir)
		for url in urls:
			filename = url.split("/")[-1]
			target_path = os.path.join(target_dir, filename)
			pool.submit(url_download, url, target_path)

# future update:
# using rich
# inside a 'context' box:
# top pbar is for the url in urls
# inside, individual pbar for all the downloads
# see nala package on ubuntu


def main():
	url = [
		"https://www.nhc.noaa.gov/gis/hazardmaps/PR_SLOSH_MOM_Inundation.zip",
		"https://imgs.xkcd.com/comics/data_quality.png",
		"https://imgs.xkcd.com/comics/omniknot.png"
	]
	target_dist = get_download_dir()
	downloader(url, target_dist)


if __name__ == '__main__':
	main()
