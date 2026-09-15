"""Locate and load a project's configs/ directory: yaml files and its data_model.py module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import orca
import yaml

_yaml_cache: dict[tuple[Path, str], Any] = {}
_data_model_cache: dict[Path, ModuleType] = {}


def get_project_dir() -> Path:
    """The project root registered by the CLI."""
    return Path(orca.get_injectable("project_dir"))


def configs_dir(project_dir: Path | None = None) -> Path:
    return Path(project_dir or get_project_dir()) / "configs"


def load_yaml(name: str, project_dir: Path | None = None, required: bool = True) -> dict[str, Any]:
    """Read configs/<name> and cache it; returns {} for an optional file that does not exist."""
    path = configs_dir(project_dir) / name
    key = (path.parent, name)
    if key in _yaml_cache:
        return _yaml_cache[key]

    if not path.is_file():
        if required:
            raise FileNotFoundError(f"{name} not found in {path.parent}")
        _yaml_cache[key] = {}
        return _yaml_cache[key]

    _yaml_cache[key] = yaml.safe_load(path.read_text()) or {}
    return _yaml_cache[key]


def load_data_model(project_dir: Path | None = None) -> ModuleType:
    """Import the project's data_model.py, whose path comes from the `data_model` key in settings.yaml."""
    directory = configs_dir(project_dir)
    if directory in _data_model_cache:
        return _data_model_cache[directory]

    settings = load_yaml("settings.yaml", project_dir)
    filename = settings.get("data_model")
    if not filename:
        raise KeyError(
            f"settings.yaml in {directory} has no 'data_model' key; it must name the project's "
            f"data model module, e.g. 'data_model: data_model.py'."
        )

    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"settings.yaml points 'data_model' at {path}, which does not exist.")

    # module name is namespaced by project so two projects' data models can coexist
    module_name = f"mazsim_data_model_{directory.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the data model module at {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for attr in ("TABLE_MODELS", "TABLE_INDEXES"):
        if not hasattr(module, attr):
            raise AttributeError(f"{path} must define {attr}.")

    _data_model_cache[directory] = module
    return module
