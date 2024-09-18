# BECQDBase.jl

**BECQDBase.jl** is a Julia module designed for simulating CQD systems using Bloch equations. It defines key structures and functions required for running experiments and simulations in this context.

## Author
**Xukun Lin**

## Last Update
09/16/2024

## Overview

The module provides several core functionalities:
- **Experiment**: Represents an experimental setup for CQD simulations.
- **Simulation**: Defines a simulation environment for running Bloch equation dynamics.
- **Result**: Contains the results of the simulation, including raw data and various plots.

In addition, the module provides functions for simulating atom dynamics, calculating magnetic fields, and visualizing results.

## Exported Constants
The following constants are essential for the CQD simulations:
- `μ₀`: Magnetic constant (vacuum permeability).
- `γₑ`: Electron gyromagnetic ratio.
- `γₙ`: Nuclear gyromagnetic ratio.

## Exported Structures and Functions
The module exports the following key components:

### Structures
- `Experiment`: Defines an experimental setup, with predefined experiments available.
- `Simulation`: Configures the simulation, including initial conditions and solvers.
- `Result`: Stores the simulation results and provides data and visualization outputs.

### Functions
- `sample_atom_once(simulation::Simulation)`: Samples one atom for the simulation.
- `sample_atoms(simulation::Simulation)`: Samples multiple atoms based on the simulation parameters.
- `is_flipped(angles::Vector, simulation::Simulation)`: Determines whether an atom has flipped based on its angles and simulation parameters.
- `get_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, simulation::Simulation)`: Computes the magnetic fields at a given time for the experiment.
- `simulate(experiment::Experiment, simulation::Simulation)`: Runs the full simulation and returns raw data and plots.
- `save_results(experiment::Experiment, simulation::Simulation, result::Result, start_time, file_dir)`: Saves the results and metadata of the simulation to files.

## Installation

To use the `BECQDBase.jl` module, you will need the following packages:

- `Pkg`, `Dates`, `LinearAlgebra`, `Statistics`, `Logging`, `StatsBase`, `DifferentialEquations`, `Plots`, `DataStructures`, `DataFrames`, `CSV`, `LaTeXStrings`, `JSON3`

You can add the required packages using the Julia package manager:

```julia
using Pkg
Pkg.add(["StatsBase", "DifferentialEquations", "Plots", "DataStructures", "DataFrames", "CSV", "LaTeXStrings", "JSON3"])
```

## Usage
Here is a basic example of how to use the module:
```julia
using Dates, BECQDBase

# Get simulation start time
start_time = now()

# Define an experiment
experiment = Experiment("04.06.2024 Alex")

# Define a simulation
simulation = Simulation(1000, "quadrupole", "up", "heart shape", RadauIIA5())

# Run the simulation
raw_data, θₑ_plot, θₙ_plot, θₑθₙ_plot = simulate(experiment, simulation)

# Save the results
save_results(experiment, simulation, result, start_time, "results_directory")
```

## Predefined Experiments
`BECQDBase.jl` comes with several predefined experiments. These can be accessed by passing the appropriate experiment name to the Experiment constructor:
- "04.06.2024 Alex"
- "04.18.2024 Alex"
- "04.23.2024 Alex"
- "08.21.2024 Alex"
- "FS Low $z_a$"
- "FS High $z_a$"

## Acknowledgement
Many parts of the code are modified based on Kelvin's code.