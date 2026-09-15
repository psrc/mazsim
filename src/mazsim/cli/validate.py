import orca
import sys
import tables
import yaml
import numpy as np
import time
from typing import Any
from pathlib import Path
import argparse

from mazsim import control_totals, data_loader, submodels, variable_loader, outputs
from mazsim.outputs import get_last_run_number, start_run_log
from mazsim.submodels import initialize_submodels
from mazsim.util import save_scatter_html


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

def _load_validate_yaml(project_dir: Path) -> dict[str, Any]:
    config_path = project_dir / "configs" / "validate.yaml"
    return yaml.safe_load(config_path.read_text())




@orca.step('create_validation_summaries')
def create_validation_summaries():
    project_dir = orca.get_injectable('project_dir')
    geographies = orca.get_injectable('validation_geographies')
    variables = orca.get_injectable('validation_variables')
    year = orca.get_injectable('year')
    obs_year_match = []
    for variable in variables:
        if orca.get_injectable(f'observed_{variable}_year') == year:
            obs_year_match.append(variable)
    if not obs_year_match:
        return

    output_dir = Path.joinpath(project_dir, 'output', 'validation_summaries')
    output_dir.mkdir(parents=True, exist_ok=True)

    total_cols = ['total_'+v for v in obs_year_match]
    obs_cols = [f'sum_obs_{v}' for v in obs_year_match]
    for table in geographies:  
        df = orca.get_table(table).to_frame(total_cols + obs_cols)
        df.to_csv(Path.joinpath(output_dir, f'{table}_{year}.csv'))
        for variable, total_col, obs_col in zip(obs_year_match, total_cols, obs_cols):
            save_scatter_html(df, table, variable, total_col, obs_col, year, output_dir)



def run(args):
    """Run the orca steps listed under preprocessing_steps in simulate.yaml."""
    start_time = time.time()

    # set up project directory and orca injectables
    project_dir = Path(args.configs_dir).parent
    orca.add_injectable("project_dir", project_dir)
    simulate_yaml = _load_simulate_yaml(project_dir)
    validate_yaml = _load_validate_yaml(project_dir)
    orca.add_injectable('validation_geographies',validate_yaml['geographies'])
    orca.add_injectable('validation_variables',validate_yaml['variables'])


    # run preprocessing steps
    additional_preprocessing_steps = validate_yaml.get('additional_preprocessing_steps',[])
    preprocessing_steps = simulate_yaml["preprocessing_steps"] + additional_preprocessing_steps
    orca.run(preprocessing_steps)

    # setup run iteration length using years from the observed data
    base_year = orca.get_injectable("base_year")
    obs_hh_year = orca.get_injectable("observed_households_year")
    obs_jobs_year = orca.get_injectable("observed_jobs_year")
    validation_year = max(obs_hh_year, obs_jobs_year)
    iter_vars = range(base_year + 1, validation_year + 1)
    
    # set the random seed for reproducibility
    np.random.seed(simulate_yaml.get("random_seed", 42))

    # run simulation steps
    additional_simulation_steps = validate_yaml.get('additional_simulation_steps',[])
    simulation_steps = simulate_yaml["simulation_steps"] + additional_simulation_steps
    orca.run(simulation_steps, iter_vars)

    # run post-processing steps
    additional_postprocessing_steps = validate_yaml.get('additional_postprocessing_steps',[])
    postprocessing_steps = simulate_yaml['postprocessing_steps'] + additional_postprocessing_steps
    orca.run(postprocessing_steps)

    # run validation summaries


    end_time = time.time()
    print(f"Simulation completed in {(end_time - start_time)/60:.2f} minutes")
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))