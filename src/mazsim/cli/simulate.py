import orca
import sys
import yaml
from typing import Any
from pathlib import Path
import argparse

from mazsim import control_totals, data_loader, variables


def _load_simulate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "simulate.yaml"
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


def run(args):
    """Run the orca steps listed under preprocessing_steps in simulate.yaml."""
    project_dir = Path(args.configs_dir).parent
    preprocessing_steps = _load_simulate_yaml(project_dir)["preprocessing_steps"]
    simulation_steps = _load_simulate_yaml(project_dir)["simulation_steps"]
    orca.add_injectable("project_dir", project_dir)
    iter_vars =_load_simulation_years(project_dir)
    orca.run(preprocessing_steps)
    orca.run(simulation_steps,iter_vars)
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))