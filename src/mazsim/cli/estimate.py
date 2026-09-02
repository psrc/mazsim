"""Estimate the household (HLCM) and job (JLCM) location choice models defined in estimate.yaml."""

import argparse
from pathlib import Path
from typing import Any
import sys

import orca
import yaml
from urbansim.models import util
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import LargeMultinomialLogitStep

# side-effect imports: registers the load_data/build_networks/register_variables orca steps
from mazsim import data, variables


def _load_estimate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "estimate.yaml"
    return yaml.safe_load(config_path.read_text())


def _load_config(project_dir: Path, key: str) -> dict[str, Any]:
    return _load_estimate_yaml(project_dir)[key]


def _fit_submodel(config: dict[str, Any], model_config: dict[str, Any]) -> None:
    """Fit one location choice sub-model (household, job, or housing unit) and register it with modelmanager."""
    filter_value = model_config["filter_value"]

    m = LargeMultinomialLogitStep()
    m.choosers = [config["choosers"]]
    m.chooser_filters = config["chooser_filters_template"].format(value=filter_value)
    m.alternatives = [config["alternatives"]]
    m.choice_column = config["choice_column"]
    m.constrained_choices = config["constrained_choices"]
    m.alt_sample_size = config["alt_sample_size"]
    m.out_chooser_filters = config["out_chooser_filters_template"].format(value=filter_value)
    m.alt_capacity = config["alt_capacity"]
    m.out_alt_filters = config["out_alt_filters"]

    m.model_expression = util.str_model_expression(config["expl_vars"], add_constant=False)
    m.fit()
    m.name = model_config["name"]
    mm.register(m)  # overwrites any previously registered model of the same name


@orca.step("estimate_hlcm")
def estimate_hlcm(project_dir: Path) -> None:
    """Fit and register every HLCM sub-model listed in estimate.yaml."""
    mm.initialize(Path.joinpath(project_dir, "configs"))
    config = _load_config(project_dir, "hlcm")

    for model_config in config["models"]:
        _fit_submodel(config, model_config)


@orca.step("estimate_jlcm")
def estimate_jlcm(project_dir: Path) -> None:
    """Fit and register every JLCM sub-model listed in estimate.yaml."""
    mm.initialize(Path.joinpath(project_dir, "configs"))
    config = _load_config(project_dir, "jlcm")

    for model_config in config["models"]:
        _fit_submodel(config, model_config)


@orca.step("estimate_hulcm")

def estimate_hulcm(project_dir: Path) -> None:
    """Fit and register every HULCM sub-model listed in estimate.yaml."""
    mm.initialize(Path.joinpath(project_dir, "configs"))
    config = _load_config(project_dir, "hulcm")

    for model_config in config["models"]:
        _fit_submodel(config, model_config)


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )

def run(args):
    """Run the orca steps listed under estimation_steps in estimate.yaml."""
    project_dir = Path(args.configs_dir).parent
    estimation_steps = _load_estimate_yaml(project_dir)["estimation_steps"]
    orca.add_injectable("project_dir", project_dir)
    orca.run(estimation_steps)
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))