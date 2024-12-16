# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from sphinxawesome_theme import ThemeOptions
from dataclasses import asdict

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

project = 'torch-ttt'
copyright = '2024, Nikita Durasov'
author = 'Nikita Durasov'
# html_title = "torch<span style='border-radius: 4px; color: white; text-justify: none; padding: 0px 2px 0px 2px; background: linear-gradient(45deg , #8c52ff, #ff914d); -moz-linear-gradient(45deg , #8c52ff, #ff914d); -webkit-linear-gradient(45deg , #8c52ff, #ff914d);'>-ttt</span>"

html_title = "torch<span style='border-radius: 4px; color: white; text-justify: none; padding: 0px 2px 0px 2px; background: linear-gradient(45deg , rgba(140, 82, 255, 0.3), rgba(255, 145, 77, 0.3)); -moz-linear-gradient(45deg , rgba(140, 82, 255, 0.3), rgba(255, 145, 77, 0.3)); -webkit-linear-gradient(45deg , rgba(140, 82, 255, 0.3), rgba(255, 145, 77, 0.3));'>-ttt</span>"

html_favicon = "_static/images/torch-ttt.svg"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    'sphinx.ext.autosummary',
    'sphinx_gallery.gen_gallery'
    # 'autoapi.extension'
    # "extend_parent",
    # "sphinx_design",
    # "myst_nb",
]

autosummary_generate = True
# autoapi_dirs = ['../torch_ttt']

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'download_all_examples': False,
     "show_signature": False
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinxawesome_theme'
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
    'css/docstring_custom.css'
]

html_js_files = [
    'js/theme.js'
]

autosummary_generate = True  # Automatically generate stub `.rst` files

pygments_style_dark = "github-dark"

html_logo = "_static/images/torch-ttt.svg"
# html_logo = "_static/images/auto_awesome.svg"

theme_options = ThemeOptions(
    # mode="dark",  # Enforce dark mode here
    extra_header_link_icons={
        "repository on GitHub": {
            "link": "https://github.com/nikitadurasov/torch-ttt",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        },
    },
)

html_theme_options = asdict(theme_options)

pygments_style = 'zenburn'
