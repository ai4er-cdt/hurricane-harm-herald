from __future__ import annotations

from pathlib import Path
from typing import List

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax


def convert_bytes(size: float) -> str:
	"""Function to convert bytes into a human-readable format

	Parameters
	----------
	size : float
		size in bytes to be converted.

	Returns
	-------
	str
		string of the converted size.

	References
	----------
	https://stackoverflow.com/a/59174649/9931399
	"""
	for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
		if size < 1024.0:
			return f"{size:3.2f} {x}"
		size /= 1024.0
	return str(size)


def rich_table(args, values, title):
	console = Console()
	table = Table(
		title=title,
		show_header=True,
		header_style="bold magenta"
	)
	table.add_column("Variable")
	table.add_column("Value")

	for i in args:
		table.add_row(
			i,
			Syntax(str(values[i]), lexer="python")
		)

	console.print(table)


def pad_number_with_zeros(
		number: str | int
) -> str:
	"""
	Add a leading zero to any number, X, into 0X. Useful for generating dates in URL strings.
	"""
	if not type(number) == str:
		try:
			number = str(number)
		except ValueError:
			print(f'Failed to convert {number} to string')
	if len(number) == 1:
		number = ''.join(('0', number))

	return number


def check_files_in_list_exist(
		file_list: List[str] | List[Path]
) -> list:
	"""State which files don't exist and remove from list."""
	files_found = []
	for fl in file_list:
		# attempt conversion to Path object if necessary
		if type(fl) != Path:
			try:
				fl = Path(fl)
			except TypeError:
				print(f"{fl} could not be converted to Path object")

		if fl.is_file():
			files_found += fl,  # TODO: check this with append
		else:
			print(f"{fl} not found. Removing from list.")

	return files_found
