import sys
import os
# import h3
sys.path.append(os.path.abspath("../.."))
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hurricane Harm Herald - H3'
copyright = '2023, Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng'
author = 'Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.viewcode',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
	# 'sphinx.ext.autosummary',
	'sphinx.ext.mathjax',
	'sphinx-mathjax-offline',
	]

templates_path = ['_templates']
exclude_patterns = []

# SPHINX_APIDOC_OPTIONS=members,show-inheritance
# add_module_names = False
# autosummary_generate = True

autodoc_mock_imports = [
	"affine",
	"cartopy",
	"cdsapi",
	"cv2",
	"geopandas",
	"geopy",
	"matplotlib",
	"numpy",
	"numba",
	"pandas",
	"PIL",
	"pytorch_lightning",
	"rasterio",
	"rich",
	"richdem",
	"shapely",
	"sklearn",
	"seaborn",
	"scipy",
	"torch",
	"torchvision",
	"torchmetrics",
	"tqdm",
	"xarray"
	]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
