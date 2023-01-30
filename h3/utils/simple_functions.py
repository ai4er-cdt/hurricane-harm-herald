def convert_bytes(size):
	"""Function to convert bytes into a human-readable format
	https://stackoverflow.com/a/59174649/9931399
	"""
	for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
		if size < 1024.0:
			return "%3.1f %s" % (size, x)
		size /= 1024.0
	return size
