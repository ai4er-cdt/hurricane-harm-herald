from __future__ import annotations

import json
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
from h3.utils.directories import get_download_dir, get_data_dir

CHUNK_SIZE = 1024


# done_event = Event()
#
#
# def handle_signals(signum, handle):
# 	done_event.set()
#
#
# signal.signal(signal.SIGINT, handle_signals)


def _credential_helper(base_url: str) -> tuple[str, str]:
	"""Getting credentials from a file, and generating them if it does not exist"""

	credential_path = os.path.join(get_data_dir(), "credentials.json")
	cred = {}

	if os.path.exists(credential_path):
		with open(credential_path, "r") as f:
			cred = json.load(f)

	if base_url not in cred:
		print(f"Credential for {base_url}")
		username = str(input("Username: "))
		password = str(input("Password: "))
		cred[base_url] = {"username": username, "password": password}
		with open(credential_path, "w") as f:
			json.dump(cred, f)
	else:
		username = cred[base_url]["username"]
		password = cred[base_url]["password"]

	return username, password


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


def _get_response(url: str) -> HTTPResponse:
	try:
		response = urllib.request.urlopen(url)
	except urllib.error.HTTPError:
		import base64
		from http.cookiejar import CookieJar

		cj = CookieJar()
		opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
		request = urllib.request.Request(url)

		user, password = _credential_helper(base_url=os.path.dirname(url))

		base64string = base64.b64encode((user + ":" + password).encode("ascii"))
		request.add_header("Authorization", f"Basic {base64string.decode('ascii')}")
		response = opener.open(request)
	except urllib.error.URLError:
		# work around to be able to dl the 10m coastline without issue
		import ssl
		ssl._create_default_https_context = ssl._create_unverified_context
		req = urllib.request.Request(url)
		req.add_header(
			"user-agent",
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
		)
		response = urllib.request.urlopen(req)
	return response


def url_download(url: str, path: str, task: int = 1, total: int = 1) -> None:
	"""
	Download an url to a local file

	See Also
	--------
	downloader : Downloads multiple url in parallel.
	"""
	logger.info(f"Downloading: '{url}' to {path}")
	response = _get_response(url)
	chunks = _get_chunks(response)
	pbar = tqdm(
		desc=f"[{task}/{total}] Requesting {os.path.basename(url)}",
		unit="B",
		total=_get_response_size(response),
		unit_scale=True,
		# format to have current/total size with the full unit, e.g. 60kB/6MB
		# https://github.com/tqdm/tqdm/issues/952
		bar_format="{l_bar}{bar}| {n_fmt}{unit}/{total_fmt}{unit} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
	)
	with pbar as t:
		with open(path, "wb") as file:
			for chunk in chunks:
				file.write(chunk)
				t.update(len(chunk))
			# if done_event.is_set():
			# 	return
	logger.debug(f"Downloaded in {path}")


def downloader(urls: Iterable[str], target_dir: str = get_download_dir()):
	"""
	Downloader to download multiple files.
	"""
	with ThreadPoolExecutor(max_workers=4) as pool:
		target_dir = os.path.abspath(target_dir)
		for task, url in enumerate(urls, start=1):
			filename = url.split("/")[-1]
			target_path = os.path.join(target_dir, filename)
			pool.submit(url_download, url, target_path, task, total=len(urls))


# future update:
# using rich
# inside a 'context' box:
# top pbar is for the url in urls
# inside, individual pbar for all the downloads
# see nala package on ubuntu


def main():
	url = [
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N29W096.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N30W096.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N30W086.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N33W080.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N33W079.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N34W079.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N34W078.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N18W075.zip',
		'https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/ASTGTMV003_N18W074.zip',
	]
	response = _get_response(url[0])
	print(response)

	target_dist = get_download_dir()
	downloader(url, target_dist)


if __name__ == "__main__":
	main()
