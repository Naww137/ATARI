# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'atari'
copyright = '2024, Noah A.W. Walton'
author = 'Noah A.W. Walton'
release = '1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.napoleon',
'sphinx.ext.doctest',
'sphinx.ext.intersphinx',
'sphinx.ext.todo',
# 'sphinx.ext.coverage',
# 'sphinx.ext.ifconfig',
'sphinx.ext.viewcode',
#    'sphinx.ext.imgmath',
'sphinx.ext.mathjax',
# 'numpydoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'
# Whether to create a Sphinx table of contents for the lists of class methods and attributes. If a table of contents is made, Sphinx expects each entry to have a separate page. True by default
numpydoc_class_members_toctree = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_permalinks_icon = '<span>#</span>'
# html_theme = 'sphinxawesome_theme'
# html_theme = 'alabaster'
# html_static_path = ['_static']
# html_theme = 'cloud'
# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'members': True,            # Include module-level members (functions, variables, etc.)
    'autodoc-skip-member': 'attribute',
    # 'member-order': 'bysource', # Order members by the source order
    # 'special-members': '__init__',  # Include special members like __init__
    # 'undoc-members': True,      # Include members without docstrings
    # 'show-inheritance': True,   # Show inheritance information
    'noindex': True,            # Do not generate a separate page for each attribute
}

autosummary_generate = True
napoleon_use_ivar = True
