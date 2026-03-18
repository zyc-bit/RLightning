# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import datetime
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

# =============================================================================
# Mock heavy dependencies for autodoc
# Uses a custom module finder that intercepts all submodule imports automatically
# =============================================================================

MOCK_MODULES = [
    "codetiming",
    "easydict",
    "gymnasium",
    "hydra",
    "imageio",
    "intervaltree",
    "joblib",
    "json_numpy",
    "loop_rate_limiters",
    "lxml",
    "mani_skill",
    "matplotlib",
    "mink",
    "mujoco",
    "natsort",
    "numpy",
    "omegaconf",
    "open3d",
    "PIL",
    "pybase64",
    "pydantic",
    "ray",
    "rich",
    "scipy",
    "smpl_sim",
    "smplx",
    "tensordict",
    "tensorflow",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
    "tree",
    "yaml",
    "zmq",
]


class _MockBase:
    """Empty base class for mocked class inheritance."""

    pass


class MockModule(ModuleType):
    """A mock module that returns itself for any attribute access."""

    def __init__(self, name=""):
        super().__init__(name)
        self.__all__ = []
        self.__file__ = ""
        self.__path__ = []

    def __repr__(self):
        return f"MockModule({self.__name__!r})"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        mock = MockModule(f"{self.__name__}.{name}" if self.__name__ else name)
        setattr(self, name, mock)
        return mock

    def __call__(self, *args, **kwargs):
        return MockModule(self.__name__)

    # Support type union syntax: str | MockModule
    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    # Make it subscriptable for generics: MockModule[T]
    def __getitem__(self, item):
        return self

    # For class inheritance: class Foo(nn.Module) -> class Foo(_MockBase)
    def __mro_entries__(self, bases):
        return (_MockBase,)


class MockLoader(Loader):
    """Loader that creates MockModule instances."""

    def create_module(self, spec):
        return MockModule(spec.name)

    def exec_module(self, module):
        pass  # Nothing to execute


class MockModuleFinder(MetaPathFinder):
    """A module finder that mocks specified modules and all their submodules."""

    def __init__(self, mock_modules):
        self.mock_modules = set(mock_modules)
        self.loader = MockLoader()

    def find_spec(self, fullname, path, target=None):
        # Check if this module or any parent is in mock list
        parts = fullname.split(".")
        for i in range(len(parts)):
            if ".".join(parts[: i + 1]) in self.mock_modules:
                return ModuleSpec(fullname, self.loader)
        return None


# Install the mock finder BEFORE adding project to path
sys.meta_path.insert(0, MockModuleFinder(MOCK_MODULES))

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath("../.."))

# Pre-import rlightning now while mocks are active
# This ensures it's cached in sys.modules before sphinx does anything else
import rlightning  # noqa: E402

# Also set autodoc_mock_imports as fallback
autodoc_mock_imports = MOCK_MODULES

project = "RLightning"
author = "RLightning Contributors"
release = "0.1.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "algorithms/**",
]

html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Show a nice footer year range by default.
html_last_updated_fmt = "%Y-%m-%d"

# Optional metadata for OpenGraph or custom themes (safe to ignore).
html_title = f"{project} Documentation"

# Keep autosectionlabel unique across the project
autosectionlabel_prefix_document = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Stable build timestamp for deterministic builds in RTD previews
today = str(datetime.date.today())
