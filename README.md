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
