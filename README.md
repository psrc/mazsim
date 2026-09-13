# mazsim

## Installation
1. install UV package manager
2. install Microsoft Visual Studio Community and make sure the C++ MSVC build tools option is selected during install
3. Setup pandana:
    - manually clone the jkolberg pandana fork from github.com/jkolberg/pandana and switch to the pandas_23 branch
    - change the tool.uv.sources path in pyproject.toml to your cloned pandana directory location
4. Place base year data in projects/baseline_summer2026/data
5. create the uv venv:

    ```uv sync```

6. run estimation with the following command:
    
    ```uv run mazsim estimate -c projects\baseline_summer2026\configs```

7. run calibration with the following command:
    
    ```uv run mazsim calibrate -c projects\baseline_summer2026\configs```

8. run validation with the following command:

    ```uv run mazsim validate -c projects\baseline_summer2026\configs```

    validation runs the simulation out the most recent year that's included in the observed_data table. The simulation can then be compared to the observerd data before running the full simulation.