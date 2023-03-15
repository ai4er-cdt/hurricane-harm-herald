import h3
from h3.utils.file_ops import check_all_downloads


def main():
	check_all_downloads()
	h3.logger.info("Getting args")
	args = h3.parser
	print(args)


if __name__ == "__main__":
	main()
