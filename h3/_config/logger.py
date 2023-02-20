"""
The logger can be accessed with `h3.logger`, or with `logging.getLogger("h3")
once `logging`has been imported.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def make_logger(level: str = "INFO") -> logging.Logger:
	FORMAT = "%(message)s"
	logging.basicConfig(
		level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
	)

	logger = logging.getLogger("h3")
	logger.setLevel(level)
	return logger
