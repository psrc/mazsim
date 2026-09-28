"""Estimate the location choice (HLCM/JLCM/HULCM) and price (HUPM) models defined in estimate.yaml."""

import argparse
from pathlib import Path
from typing import Any
import sys

import orca
import yaml
from urbansim.models import util
from urbansim_templates import modelmanager as mm
from urbansim_templates.models import LargeMultinomialLogitStep, OLSRegressionStep

# side-effect imports: registers the load_data/build_networks/register_variables orca steps
from mazsim import data_loader, variable_loader
from mazsim.submodels import initialize_submodels


def _load_estimate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "estimate.yaml"
    return yaml.safe_load(config_path.read_text())


def _load_calibration_period(project_dir: Path) -> tuple[int, int]:
    """The base/historic years from calibrate.yaml, which fill the filter template placeholders."""
    config_path = project_dir / "configs" / "calibrate.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found; it supplies base_year and historic_year, which the "
            "sub-model filter templates are filled from."
        )
    calibrate_yaml = yaml.safe_load(config_path.read_text())
    return calibrate_yaml["base_year"], calibrate_yaml["historic_year"]


def _load_config(project_dir: Path, key: str) -> dict[str, Any]:
    return _load_estimate_yaml(project_dir)[key]


def _fit_submodel(
    config: dict[str, Any], model_config: dict[str, Any], base_year: int, historic_year: int
) -> None:
    """Fit one location choice sub-model (household, job, or housing unit) and register it with modelmanager."""
    filter_value = model_config["filter_value"]
    period = {"base_year": base_year, "historic_year": historic_year}

    m = LargeMultinomialLogitStep()
    m.choosers = [config["choosers"]]
    m.chooser_filters = config["chooser_filters_template"].format(value=filter_value, **period)
    m.alternatives = [config["alternatives"]]
    m.choice_column = config["choice_column"]
    m.constrained_choices = config["constrained_choices"]
    m.alt_sample_size = config["alt_sample_size"]
    m.out_chooser_filters = config["out_chooser_filters_template"].format(value=filter_value, **period)
    m.alt_capacity = config["alt_capacity"]
    m.out_alt_filters = config["out_alt_filters"]

    m.model_expression = util.str_model_expression(config["expl_vars"], add_constant=False)
    m.fit()
    m.name = model_config["name"]
    mm.register(m)  # overwrites any previously registered model of the same name


def _fit_hupm_submodel(config: dict[str, Any], model_config: dict[str, Any]) -> None:
    """Fit one OLS price sub-model (residential value or rent) and register it with modelmanager."""
    m = OLSRegressionStep()
    m.tables = config["tables"]
    m.filters = model_config["filters"]

    model_spec = {"left_side": model_config["left_side"], "right_side": config["expl_vars"]}
    m.model_expression = util.str_model_expression(model_spec)
    m.out_filters = config["out_filters"]
    m.out_column = model_config["left_side"]
    m.fit()
    m.name = model_config["name"]
    mm.register(m)  # overwrites any previously registered model of the same name


@orca.step("estimate_hlcm")
def estimate_hlcm(project_dir: Path) -> None:
    """Fit and register every HLCM sub-model listed in estimate.yaml."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hlcm")
    base_year, historic_year = _load_calibration_period(project_dir)

    for model_config in config["models"]:
        _fit_submodel(config, model_config, base_year, historic_year)


@orca.step("estimate_jlcm")
def estimate_jlcm(project_dir: Path) -> None:
    """Fit and register every JLCM sub-model listed in estimate.yaml."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "jlcm")
    base_year, historic_year = _load_calibration_period(project_dir)

    for model_config in config["models"]:
        _fit_submodel(config, model_config, base_year, historic_year)


@orca.step("estimate_hulcm")

def estimate_hulcm(project_dir: Path) -> None:
    """Fit and register every HULCM sub-model listed in estimate.yaml."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hulcm")
    base_year, historic_year = _load_calibration_period(project_dir)

    for model_config in config["models"]:
        _fit_submodel(config, model_config, base_year, historic_year)


@orca.step("estimate_hupm")
def estimate_hupm(project_dir: Path) -> None:
    """Fit and register every HUPM sub-model listed in estimate.yaml."""
    initialize_submodels(project_dir)
    config = _load_config(project_dir, "hupm")

    for model_config in config["models"]:
        _fit_hupm_submodel(config, model_config)


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