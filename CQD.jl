include("CQDBase.jl")
using Alert, Dates, .CQDBase

file_dir = pwd()

# User Control
start_time = now()
experiment = Experiment("Alex 165")
simulation = Simulation(
    "BE", # type
    15, # number of atoms
    "exact", # field
    "down", # initial μₑ
    "Iso", # initial μₙ
    "radau5", # solver
    true, # θₙ is fixed
    "Bₑ dominant", # branching condition
    "CQD", # BₙBₑ strength
    (1, 1), # Bₙ Bₑ ratio
    0, # kᵢ
    ("ABC", 1/16), # average method
    "off", # θ cross detection
    "off", # sigmoid field
    "qm", # R2 comparison
)
raw_data, θₑ_plot, θₙ_plot, θₑθₙ_plot = simulate(experiment, simulation)
results = Results(experiment, simulation, raw_data, θₑ_plot, θₙ_plot, θₑθₙ_plot)
save_results(experiment, simulation, results, start_time, file_dir)
cp(@__FILE__, joinpath(file_dir, Dates.format(start_time, "yyyy-mm-dd_HH-MM-SS-sss"), "Code.jl"))
alert("Simulation finished!")