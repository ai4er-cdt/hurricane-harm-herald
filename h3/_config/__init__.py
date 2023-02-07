from __future__ import annotations

import logging

from .logger import make_logger


logger: logging.Logger

logger = make_logger(
	level="DEBUG"
)
