import os
import sys
import re

# If you installed the package in editable mode (`uv sync`), Sphinx can often find it.
# However, pointing explicitly to 'src' is the most reliable method.
sys.path.insert(0, os.path.abspath('../../src'))

def get_project_metadata():
    import pathlib
    import sys
    pyproject_path = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    if sys.version_info >= (3, 11):
        import tomllib
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    else:
        try:
            import tomli
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
        except ImportError:
            return {}
    project = data.get("project", {})
    author = project.get("authors", [{}])[0].get("name", "")
    copyright_year = re.search(r"\\d{4}", project.get("version", ""))
    copyright_str = f"{copyright_year.group(0) if copyright_year else ''}, {author}"
    return {
        "project": project.get("name", "SheetWise"),
        "author": author,
        "release": project.get("version", "0.0.0"),
        "copyright": copyright_str,
    }

meta = get_project_metadata()

project = "Tenso"
copyright = meta.get("copyright", "2025, Khushiyant")
author = meta.get("author", "Khushiyant")
release = meta.get("release", "0.6.0")

# Mock heavy dependencies so docs can build without installing CUDA/Torch
autodoc_mock_imports = [
    "cupy", 
    "torch", 
    "numpy"  # Optional: speeds up build if you don't need numpy internals
    "xxhash"
]

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation
    'sphinx.ext.napoleon',     # Parses the Google-style args in your code
    'sphinx.ext.viewcode',     # Adds "View Source" links
    'sphinx_autodoc_typehints', # Uses your python type hints automatically
    'myst_parser'  
]

templates_path = ['_templates']
exclude_patterns = []

# Theme configuration
html_static_path = ['_static']
html_theme = 'furo'

# Optional: Furo specific customization
html_theme_options = {
    "source_repository": "https://github.com/Khushiyant/tenso",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
myst_heading_anchors = 3