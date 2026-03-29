# Configuration file for the Sphinx documentation builder.

project = "Puddin"
copyright = "2026, The Puddin Authors"
author = "The Puddin Authors"
release = "0.1.0"

extensions = [
    "sphinx_rust",
    "myst_parser",
    "sphinxcontrib.katex",
]

# sphinx-rust: paths to Cargo.toml of each crate to document
rust_crates = [
    "../crates/puddin",
]

# MyST
myst_enable_extensions = ["dollarmath", "amsmath"]

# HTML theme
html_theme = "furo"
html_title = "Puddin"

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}
