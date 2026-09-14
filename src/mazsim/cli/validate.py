import orca
import sys
import yaml
import numpy as np
import time
from typing import Any
from pathlib import Path
import argparse

from mazsim import control_totals, data_loader, submodels, variable_loader
from mazsim.outputs import get_last_run_number, start_run_log
from mazsim.submodels import initialize_submodels


def add_run_args(parser):
    parser.add_argument(
        "-c",
        "--configs_dir",
        type=str,
        metavar="PATH",
        help="path to configs dir",
    )


def _load_simulate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "simulate.yaml"
    return yaml.safe_load(config_path.read_text())


def run(args):
    """Run the orca steps listed under preprocessing_steps in simulate.yaml."""
    # set up project directory and orca injectables
    project_dir = Path(args.configs_dir).parent
    orca.add_injectable("project_dir", project_dir)
    run_number = get_last_run_number(project_dir) + 1
    orca.add_injectable("run_number", run_number)
    start_run_log(project_dir, run_number)
    initialize_submodels(project_dir)
    start_time = time.time()

    simulate_yaml = _load_simulate_yaml(project_dir)
    preprocessing_steps = simulate_yaml["preprocessing_steps"]
    orca.run(preprocessing_steps)
    base_year = orca.get_injectable("base_year")
    obs_hh_year = orca.get_injectable("observed_hh_year")
    obs_jobs_year = orca.get_injectable("observed_jobs_year")
    validation_year = max(obs_hh_year, obs_jobs_year)
    iter_vars = range(base_year + 1, validation_year + 1)
    simulate_yaml = _load_simulate_yaml(project_dir)
    
    # set the random seed for reproducibility
    np.random.seed(simulate_yaml.get("random_seed", 42))

    # run simulation steps
    simulation_steps = simulate_yaml["simulation_steps"]
    orca.run(simulation_steps, iter_vars)

    end_time = time.time()
    print(f"Simulation completed in {(end_time - start_time)/60:.2f} minutes")
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))