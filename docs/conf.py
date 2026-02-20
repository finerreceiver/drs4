copyright = "2024-2026 Akio Taniguchi"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
html_static_path = ["_static"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/finerreceiver/drs4",
    "logo": {"text": "DRS4"},
    "navbar_end": [
        "version-switcher",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "switcher": {
        "json_url": "https://finerreceiver.github.io/drs4/_static/switcher.json",
        "version_match": "0.1.0",
    },
}
