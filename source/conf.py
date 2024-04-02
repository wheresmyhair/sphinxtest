# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LMFlow'
copyright = '2024, LMFlow'
author = 'LMFlow'
release = ''

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ['custom.css',]
html_logo = "_static/logo.png"
html_theme_options = {
    "announcement": "We've released our memory-efficient finetuning algorithm LISA, check out [<a href='https://arxiv.org/pdf/2403.17919.pdf'>Paper</a>][<a href='https://github.com/OptimalScale/LMFlow#finetuning-lisa'>User Guide</a>] for more details!",
    "back_to_top_button": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"]
}
