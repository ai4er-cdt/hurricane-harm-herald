import os
import importlib
import inspect


def get_h3_dir():
	h3_module = importlib.import_module("hurricane_harm_herald")
	h3_dir = os.path.dirname(inspect.getabsfile(h3_module))
	return os.path.abspath(os.path.join(h3_dir, ".."))
