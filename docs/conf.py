from __future__ import annotations

import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

project = "FG-Vector Sampler"
author = "DFTandQC"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
autodoc_typehints = "description"
