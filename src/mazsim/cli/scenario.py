import sys
import tempfile
from pathlib import Path
import argparse

import orca
import yaml

from mazsim.util import scenario_helpers
from mazsim.cli.simulate import add_run_args, run_simulation


def run(args):
    """Run a simulation with the scenario's configs and data layered over its baseline project."""
    scenario_configs = Path(args.configs_dir).resolve()
    scenario_dir = scenario_configs.parent

    scenario_settings = scenario_helpers.read_scenario_settings(scenario_configs)
    if not scenario_settings.get("baseline"):
        sys.exit(f"{scenario_configs / 'settings.yaml'} must set 'baseline' to the baseline project folder.")
    baseline_dir = scenario_helpers.resolve_baseline_dir(scenario_dir, scenario_settings["baseline"])

    baseline_settings = yaml.safe_load((baseline_dir / "configs" / "settings.yaml").read_text())
    orca.add_injectable("baseline_archive_dir", baseline_dir / baseline_settings.get("output_dir", "output") / "archive")
    if scenario_settings.get("baseline_run_number") is not None:
        orca.add_injectable("baseline_run_number", int(scenario_settings["baseline_run_number"]))

    with tempfile.TemporaryDirectory(prefix="mazsim_scenario_", ignore_cleanup_errors=True) as tmp:
        merged_configs = Path(tmp) / "configs"
        scenario_helpers.build_merged_configs(scenario_dir, baseline_dir, merged_configs)
        orca.add_injectable("configs_dir", merged_configs)
        orca.add_injectable("scenario_configs_dir", scenario_configs)
        run_simulation(scenario_dir, extra_postprocessing=["save_baseline_comparison"])
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args()
    sys.exit(run(args))
