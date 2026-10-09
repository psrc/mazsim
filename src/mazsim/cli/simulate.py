import orca
import sys
import yaml
from typing import Any
import time
import numpy as np
from pathlib import Path
import argparse

from mazsim import control_totals, data_loader, submodels, variable_loader, config
from mazsim.outputs import get_last_run_number, start_run_log
from mazsim.submodels import initialize_submodels
from mazsim.util import scenario_helpers


def _load_simulate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = config.configs_dir(project_dir) / "simulate.yaml"
    return yaml.safe_load(config_path.read_text())

def _load_observed_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = config.configs_dir(project_dir) / "observed_data.yaml"
    return yaml.safe_load(config_path.read_text())

def _load_simulation_years(project_dir: Path):
    if orca.get_injectable('run_years'):
        iter_vars = orca.get_injectable('run_years')
    else:
        base_year = orca.get_injectable('base_year')
        end_year = orca.get_injectable('end_year')
        iter_vars = range(base_year + 1, end_year + 1)
    return iter_vars


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )


def run_simulation(project_dir: Path, extra_postprocessing: list[str] | None = None) -> None:
    """Run the steps in simulate.yaml; extra steps go right before archive_results, or last without it."""
    start_time = time.time()

    # set up project directory and orca injectables
    orca.add_injectable("project_dir", project_dir)
    simulate_yaml = _load_simulate_yaml(project_dir)
    observed_yaml = _load_observed_yaml(project_dir)

    postprocessing_steps = list(simulate_yaml['postprocessing_steps'])
    if extra_postprocessing:
        position = postprocessing_steps.index('archive_results') if 'archive_results' in postprocessing_steps else len(postprocessing_steps)
        postprocessing_steps[position:position] = extra_postprocessing

    # run preprocessing steps
    orca.run(simulate_yaml["preprocessing_steps"])

    # setup run iteration length using years from the observed data
    # start with min observed year
    obs_year = min(
        orca.get_injectable(f"observed_{variable}_year")
        for variable in observed_yaml["types_to_place"]
    )
    orca.add_injectable('start_year', obs_year)
    end_year = orca.get_injectable("end_year")
    iter_vars = range(obs_year, end_year + 1)
    
    # set the random seed for reproducibility
    np.random.seed(simulate_yaml.get("random_seed", 42))

    # run simulation steps
    orca.run(simulate_yaml["simulation_steps"], iter_vars)

    # run post-processing steps
    orca.run(postprocessing_steps)

    end_time = time.time()
    print(f"Simulation completed in {(end_time - start_time)/60:.2f} minutes")


def run(args):
    """Run the orca steps listed in simulate.yaml."""
    configs_dir = Path(args.configs_dir)
    if scenario_helpers.read_scenario_settings(configs_dir).get("baseline"):
        sys.exit(f"{configs_dir} is a scenario (its settings.yaml has a 'baseline'); "
                 f"run it with: mazsim scenario -c {configs_dir}")

    run_simulation(configs_dir.parent)
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))