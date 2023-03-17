# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Hurricane Harm Herald - H3'
copyright = '2023, Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng'
author = 'Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import sys, os
sys.path.append(os.path.abspath("../.."))


extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.viewcode',
	'sphinx.ext.napoleon',
	'sphinx.ext.mathjax',
	'sphinx-mathjax-offline',
	]



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
