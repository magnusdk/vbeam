import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "vbeam"
copyright = "2023, Magnus Dalen Kvalevåg"
author = "Magnus Dalen Kvalevåg"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Generates documentation from docstrings
    "sphinx.ext.napoleon",  # Adds support for NumPy and Google style docstrings
    "sphinx.ext.autosummary",  # Automatically generates documentation for modules
    "sphinx.ext.viewcode",
    "sphinx_copybutton",  # Adds copy-button to code-blocks
    "myst_nb",  # Rendering Jupyter notebooks

    # Only works on read-the-docs
    # "sphinxcontrib.jquery",  # Adds jQuery (needed by hoverxref)
    # "hoverxref.extension",  # Hover over references
]

autodo_mock_imports = ["vbeam.core"]
autodoc_member_order = "bysource"
# autosummary_generate = True
autosummary_ignore_module_all = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/vbeam_header.png"
html_favicon = "_static/favicon-92x92.png"


hoverxref_roles = [
    "term",
]
hoverxref_role_types = {
    "term": "tooltip",
}