def convert_bytes(size: float) -> str:
	"""Function to convert bytes into a human-readable format
	https://stackoverflow.com/a/59174649/9931399

	Parameters
	----------
	size : float
		size in bytes to be converted.

	Returns
	-------
	str
		string of the converted size.
	"""
	for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
		if size < 1024.0:
			return f"{size:3.2f} {x}"
		size /= 1024.0
	return str(size)
