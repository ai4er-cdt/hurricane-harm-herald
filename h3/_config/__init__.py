from __future__ import annotations

import logging

from .logger import make_logger
from .cli_utils import cli_parser


__all__ = [
	"logger",
	# "parser"
]

# parser = cli_parser()
logger: logging.Logger

logger = make_logger(
	level="INFO"
)
