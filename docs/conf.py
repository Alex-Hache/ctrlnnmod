# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ctrlnmod'
copyright = '2025, Alexandre Hache'
author = 'Alexandre Hache'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',           # Google/NumPy docstrings
    'sphinx.ext.autosummary',        # Summary tables
    'sphinx_autodoc_typehints',      # Uses Python type hints in docs
    'myst_parser',                   # Enables Markdown support
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autosummary_generate = True
autodoc_typehints = "description"


import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # One level up to import your code
