# mazsim

## Installation
1. Install UV package manager
2. Install Microsoft Visual Studio Community and make sure the C++ MSVC build tools option is selected during install
3. Setup pandana:
    - manually clone the jkolberg pandana fork from github.com/jkolberg/pandana and switch to the pandas_23 branch
    - change the tool.uv.sources path in pyproject.toml to your cloned pandana directory location
4. Place base year data in projects/baseline_summer2026/data
5. Create the uv venv:

    ```uv sync```

6. Run estimation with the following command:
    
    ```uv run mazsim estimate -c projects\baseline_summer2026\configs```

7. Run calibration with the following command:
    
    ```uv run mazsim calibrate -c projects\baseline_summer2026\configs```

8. Run validation with the following command:

    ```uv run mazsim validate -c projects\baseline_summer2026\configs```

    Validation runs the simulation out the most recent year that's included in the observed_data table. The simulation can then be compared to the observerd data before running the full simulation.

8. Run simulation with the following command:

    ```uv run mazsim simulate -c projects\baseline_summer2026\configs```

    Simulation first forces observed jobs and housing_units to be placed and then begins the simulation using the most recent observed data year possible. In the example, the most recent observed jobs data is 2023 and most recent observed housing_unit data is 2025. The simulation starts in 2023 but continues to force the placement of housing_units in 2024 and 2025 while job placement switches to being simulated in 2024.
