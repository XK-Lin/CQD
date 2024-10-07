include("CQDBase.jl")
using .CQDBase, Dates, Plots
start_time = now()
experiment = Experiment("Alex 156")
simulation = Simulation(
    "BE",
    2,
    "quadrupole",
    "down",
    "HS",
    "RadauIIA5",
    true,
    "B₀ dominant",
    "CQD",
    0.0,
    ("ABC", 1/16),
    "off",
    (0.1, 2e-2)
)
n = 2^8
Bz = zeros(n)
i = 1
t_list = collect(range(experiment.time_span[1], experiment.time_span[2], n))
for t in t_list
    Bz[i] = get_external_magnetic_fields(t, 0.0, experiment, simulation)[3]
    global i += 1
end
field_plot = Plots.plot()
Plots.plot!(field_plot, t_list * 1e6, Bz, label = "\$B_z\$")
Plots.vspan!(collect(experiment.time_span) * 1e6, label = "", linecolor = :grey, fillcolor = :grey, alpha = 0.2)
Plots.xlabel!("\$t\$ [us]")
Plots.ylabel!("\$B_z\$ [T]")