"""Build a scenario's merged configs: the baseline project's configs overlaid with the scenario's."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import yaml

# settings.yaml keys that only a scenario uses; they are dropped from the merged settings
SCENARIO_SETTINGS = ("baseline", "baseline_run_number")

CUSTOM_DIR_SETTINGS = ("custom_steps_dir", "custom_variables_dir")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def read_scenario_settings(configs_dir: Path) -> dict[str, Any]:
    """The scenario-only keys (`baseline`, `baseline_run_number`) in a configs dir's settings.yaml."""
    path = Path(configs_dir) / "settings.yaml"
    settings = _read_yaml(path) if path.is_file() else {}
    return {key: settings[key] for key in SCENARIO_SETTINGS if key in settings}


def resolve_baseline_dir(scenario_dir: Path, baseline: str) -> Path:
    """The baseline project dir; a relative path is taken from the scenario project's parent folder."""
    path = Path(baseline)
    if not path.is_absolute():
        path = Path(scenario_dir).parent / path
    path = path.resolve()
    if not (path / "configs").is_dir():
        raise FileNotFoundError(f"settings.yaml 'baseline' points at {path}, which has no configs folder.")
    return path


def deep_merge(base: dict, override: dict) -> dict:
    """Merge dicts recursively; any other override value, lists included, replaces the base value."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_data_sources(base: dict, override: dict) -> dict:
    """Merge two data_sources.yaml contents, matching the list of single-key table entries by table name."""
    sources = list(base.get("data_sources") or [])
    positions = {next(iter(entry)): i for i, entry in enumerate(sources)}
    for entry in override.get("data_sources") or []:
        name = next(iter(entry))
        if name in positions:
            sources[positions[name]] = entry
        else:
            positions[name] = len(sources)
            sources.append(entry)

    merged = deep_merge(
        {k: v for k, v in base.items() if k != "data_sources"},
        {k: v for k, v in override.items() if k != "data_sources"},
    )
    merged["data_sources"] = sources
    return merged


def _absolutize_data_sources(sources: dict, data_dir: Path) -> dict:
    """Point the baseline's tables and archive at its data dir so they load from there, not the scenario's."""
    sources = dict(sources)
    sources["data_sources"] = [
        {name: (data_dir / filename).as_posix() for name, filename in entry.items()}
        for entry in sources.get("data_sources") or []
    ]
    if sources.get("data_archive"):
        sources["data_archive"] = (data_dir / sources["data_archive"]).as_posix()
    return sources


def build_merged_configs(scenario_dir: Path, baseline_dir: Path, dest: Path) -> None:
    """Write the baseline configs overlaid with the scenario's into `dest`, which must not exist.

    Top-level yaml files in both are merged key by key (data_sources.yaml by table name); every other
    scenario file, .py and submodels/*.yaml included, replaces the baseline's or is added. Baseline
    tables get absolute data paths; the scenario's keep bare filenames and load from its own data dir.
    """
    baseline_configs = Path(baseline_dir) / "configs"
    scenario_configs = Path(scenario_dir) / "configs"
    shutil.copytree(baseline_configs, dest, ignore=shutil.ignore_patterns("__pycache__"))

    baseline_settings = _read_yaml(baseline_configs / "settings.yaml")
    baseline_data_dir = Path(baseline_dir) / baseline_settings.get("data_dir", "data")
    sources_path = dest / "data_sources.yaml"
    if sources_path.is_file():
        _write_yaml(sources_path, _absolutize_data_sources(_read_yaml(sources_path), baseline_data_dir))

    for path in sorted(p for p in scenario_configs.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        relative = path.relative_to(scenario_configs)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        is_top_level_yaml = path.suffix == ".yaml" and len(relative.parts) == 1
        if not (is_top_level_yaml and target.exists()):
            shutil.copyfile(path, target)
        elif relative.name == "data_sources.yaml":
            _write_yaml(target, merge_data_sources(_read_yaml(target), _read_yaml(path)))
        else:
            _write_yaml(target, deep_merge(_read_yaml(target), _read_yaml(path)))

    settings_path = dest / "settings.yaml"
    settings = _read_yaml(settings_path)
    for key in SCENARIO_SETTINGS:
        settings.pop(key, None)
    for key in CUSTOM_DIR_SETTINGS:
        value = settings.get(key)
        if value is None:
            continue
        # a custom dir of 'configs' means the merged configs; any other dir belongs to the baseline project
        is_configs = Path(value).as_posix() == "configs"
        settings[key] = dest.as_posix() if is_configs else (Path(baseline_dir) / value).as_posix()
    _write_yaml(settings_path, settings)
