# mazsim

## Installation
1. Install UV package manager
    
    Windows (use powershell terminal):

    ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

    OSX/Linux:

    ```curl -LsSf https://astral.sh/uv/install.sh | sh```

2. [Install Microsoft Visual Studio Community](https://learn.microsoft.com/en-us/cpp/overview/acquire-msvc?view=msvc-170) and make sure the C++ MSVC build tools option is selected during install

3. Create the uv venv:

    ```uv sync```

4. Run example estimation with the following command: (can be skipped, example already has estimated submodels)
    
    ```uv run mazsim estimate -c examples\example_baseline\configs```

5. Run example calibration with the following command: (can be skipped, example already has calibrated submodels)
    
    ```uv run mazsim calibrate -c examples\example_baseline\configs```

6. Run example validation with the following command:

    ```uv run mazsim validate -c examples\example_baseline\configs```

    Validation runs the simulation out the most recent year that's included in the observed_data table. The simulation can then be compared to the observerd data before running the full simulation.

7. Run simulation with the following command:

    ```uv run mazsim simulate -c examples\example_baseline\configs```

    Simulation first forces observed jobs and housing_units to be placed and then begins the simulation using the most recent observed data year possible. In the example, the most recent observed jobs data is 2023 and most recent observed housing_unit data is 2025. The simulation starts in 2023 but continues to force the placement of housing_units in 2024 and 2025 while job placement switches to being simulated in 2024. Households are still placed by the LCMs so they will not perfectly match observed household data if provided.

8. Run a scenario with the following command:

    ```uv run mazsim scenario -c examples\example_scenario\configs```

    A scenario builds on a baseline project. The scenario's `settings.yaml` names it with `baseline` (a path, relative to the scenario project's parent folder or absolute) and optionally `baseline_run_number`, the archived baseline run to compare to (default: the latest). The baseline's configs and data are used as-is except where the scenario's `configs` folder overrides them:

    - Yaml files in both are merged key by key (nested dicts merge, lists are replaced). In `data_sources.yaml` the table lists merge by table name, so a scenario table is read from the scenario's `data` folder in place of the baseline's table of the same name; all other tables are read from the baseline's `data` folder.
    - `.py` files and `submodels\*.yaml` files replace the baseline's whole.
    - `scenario_data_archive` in `data_sources.yaml` is a zip of the scenario's tables, extracted automatically if any are missing.

    Results go to the scenario's `output` folder, with end-year differences from the baseline run in `output_summaries\<geography>_<end_year>_scen_minus_base.csv`. The archived run holds the merged configs in `configs` and the scenario's own files in `scenario_configs`. `simulate` refuses to run a scenario's configs.
