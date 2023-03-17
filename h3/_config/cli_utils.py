import argparse


def cli_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-z",
		"--zoom",
		choices=["0.5", "1", "2", "4"],
		type=str,
		default="1",
		help="Zoom level to use (default: %(default)s)",
	)

	parser.add_argument(
		"--model",
		type=str,
		help="Model to use",
		choices=["resnet", "satmae", "swin"]
	)

	parser.add_argument(
		"-w",
		"--weather",
		type=str,
		choices=["noaa", "ecmwf", "both"]
	)

	parser.add_argument(
		"--num-workers",
		type=int,
		default=4,
		help="Number of workers for the Dataloader. (default: %(default)s). Set to 0 if using CPU."
	)

	args = parser.parse_args()
	return args
