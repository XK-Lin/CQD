include("CQDBase.jl")
using .CQDBase, Dates, Plots

start_time = now()
experiment = Experiment("Alex 156")
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
    (0.1, 2e-2), # sigmoid field
    "qm", # R2 comparison
)
n = 2^8
Bz = zeros(n)
i = 1
t_list = collect(range(experiment.time_span[1], experiment.time_span[2], n))
for t ∈ t_list
    Bz[i] = get_external_magnetic_fields(t, 0.2, experiment, simulation)[3]
    global i += 1
end
field_plot = Plots.plot()
Plots.plot!(field_plot, t_list * 1e6, Bz, label = "\$B_z\$")
Plots.vspan!(collect(experiment.time_span) * 1e6, label = "", linecolor = :grey, fillcolor = :grey, alpha = 0.2)
Plots.xlabel!("\$t\$ [us]")
Plots.ylabel!("\$B_z\$ [T]")