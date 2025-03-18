# Add these settings to enable autodoc for all scripts

add_module_names = False     # Remove module names from generated docs
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'imported-members': False,
}
# Project information
project = 'Micaflow'
copyright = '2025, MICA Lab'
author = 'MICA Lab'

# Theme options
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 2,  # This controls dropdown depth
    'collapse_navigation': False,
    'sticky_navigation': True, 
    'titles_only': False,
}

# Logo setup
html_logo = '_static/images/logo.png'  # Add your logo here

# Static files
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
# Add the scripts directory to path so Sphinx can find them
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Root directory
sys.path.insert(0, os.path.abspath('../../scripts'))  # Scripts directory
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
]
autosummary_generate = True  # Generate stub pages for all modules
# For NumPy/Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Theme settings
html_theme = 'sphinx_rtd_theme'