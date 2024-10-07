#=
CQDBase.jl

This module defines important structures and functions for CQD simulations. These can be used for different approaches, including the original CQD simulation with BE, or the Wigner d Majorana simulation with BE.
Author: Xukun Lin
Update: 10/06/2024

Required packages: "Pkg", "LinearAlgebra", "Dates", "Statistics", "Logging", "StatsBase", "DifferentialEquations", "ODEInterfaceDiffEq", "Plots", "DataStructures", "DataFrames", "CSV", "LaTeXStrings", "JSON3", "Rotations", "WignerD".
Required constants: μ₀, γₑ, γₙ, δθ.
=#

module CQDBase

using Pkg, LinearAlgebra, Dates, Statistics, Logging, StatsBase, DifferentialEquations, ODEInterfaceDiffEq, Plots, DataStructures, DataFrames, CSV, LaTeXStrings, JSON3, Rotations, WignerD

export Experiment, Simulation, Results
export get_external_magnetic_fields, simulate, save_results

const μ₀ = 4π * 1e-7
const γₑ = -1.76085963e11
const γₙ = 1.2500612e7
const δθ = 1e-6

LinearAlgebra.BLAS.set_num_threads(4)
Plots.default(fontfamily="Computer Modern", tickfont=10, linewidth=1.5, framestyle=:box, legendfontsize=9)

"""
    struct Experiment

An `Experiment` represents the experiment to simulate.

# Fields
- `name::String`: The name of the experiment. Predefined experiment names are `"FS Low \$z_a\$"`, and `"FS High \$z_a\$"`, `"Alex 165"`, `"Alex 156"`, `"Alex 396"`, `"Alex 105"`. For predefined experiments, use `Experiment(name)`.
- `currents::Vector{Float64}`: The wire currents.
- `flip_probabilities::Vector{Float64}`: The (mean) flip probabilities from the experiment.
- `flip_probabilities_stds::Vector{Float64}`: The standard deviations of the flip probabilities.
- `qm_flip_probabilities`::Vector{Float64}: The flip probabilities according to QM simulation.
- `zₐ::Float64`: The distance from the beam to the null point.
- `v::Float64`: The velocity of the atomic beam.
- `Bᵣ::Vector{Float64}`: The remnant field.
- `system_length::Float64`: The length of the system.
- `time_span::Tuple{Float64, Float64}`: The flight time range for the atoms.
"""
struct Experiment
    name::String
    currents::Vector{Float64}
    flip_probabilities::Vector{Float64}
    flip_probabilities_stds::Vector{Float64}
    qm_flip_probabilities::Vector{Float64}
    zₐ::Float64
    v::Float64
    Bᵣ::Vector{Float64}
    system_length::Float64
    time_span::Tuple{Float64, Float64}
    function Experiment(name::String, currents::Vector{Float64}, flip_probabilities::Vector{Float64}, flip_probabilities_stds::Vector{Float64}, qm_flip_probabilities::Vector{Float64}, zₐ::Float64, v::Float64, Bᵣ::Vector{Float64}, system_length::Float64, time_span::Tuple{Float64, Float64})
        length(currents) == length(flip_probabilities) == length(flip_probabilities_stds) == length(qm_flip_probabilities) || throw(ArgumentError("The length of the current list and the flip probability list must be equal."))
        all(0 .<= flip_probabilities .<= 1) || throw(ArgumentError("Flip probability must be between 0 and 1."))
        all(isnan, flip_probabilities_stds) || all(x -> x >= 0, flip_probabilities_stds) || throw(ArgumentError("The standard deviations of the flip probabilities are invalid."))
        # all(0 .<= qm_flip_probabilities .<= 1) || throw(ArgumentError("Flip probability must be between 0 and 1."))
        zₐ > 0 || throw(ArgumentError("zₐ must be positive."))
        v > 0 || throw(ArgumentError("v must be positive."))
        system_length > 0 || throw(ArgumentError("The length of the system must be positive."))
        new(name, currents, flip_probabilities, flip_probabilities_stds, qm_flip_probabilities, zₐ, v, Bᵣ, system_length, time_span)
    end
end

"""
    Experiment(name::String)

Outer constructor for `Experiment` that takes a predefined experiment name.
"""
function Experiment(name::String)
    experiments = Dict(
        "FS Low \$z_a\$" => (-1 * [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        # [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1] / 100,
        [0.000001275588301, 0.011491017802886, 0.059235010197955, 0.147571912832191, 0.224749989856136, 0.278763884568857, 0.258084512687681, 0.159140733524888, 0.002466693134262],
        [0.000002551176603, 0.012706621114225, 0.019087441872173, 0.041843884468939, 0.065821040395060, 0.045884573918560, 0.036562847662003, 0.037966146555234, 0.004933386268525],
        [1.618181088923953e-17, 8.222328341118226e-7, 0.06413051347614414, 0.12223458274766767, 0.21266440131608902, 0.24995836392356835, 0.24999944698021592, 0.2499998869331271, 0.249999973719386],
        105e-6, 800.0, [0, 0, 42e-6], 16e-3, 16e-3 / 800 .* (-0.5, 0.5)), # German and Italian
        "FS High \$z_a\$" => (-1 * [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        # [2.4, 5.77, 12.1, 15.28, 25.38, 25.63, 17.34, 0.19] / 100,
        [0, 0.013850717454329, 0.055112422306537, 0.143874376997169, 0.221911657387476, 0.276386244805878, 0.255952792793517, 0.150871619735033, 0.002901137777317],
        [0, 0.007647700743372, 0.020866800388630, 0.037742966856351, 0.074009489126327, 0.045814684665165, 0.015520092525067, 0.025068055645921, 0.005802275554634],
        [1.618181088923953e-17, 0.010364596662094603, 0.03902217103606856, 0.07943640913244256, 0.16314288857138018, 0.24607303049227863, 0.24996591950281144, 0.24995557314677255, 0.24999341498375902],
        225e-6, 800.0, [0, 0, 42e-6], 16e-3, 16e-3 / 800 .* (-0.5, 0.5)), # German and Italian
        "Alex 165" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.0534151282288409, 0.0479013706009164, 0.103178291177038, 0.182199609641768, 0.271739445083921, 0.265583816803359, 0.251739022695542, 0.253387509472528],
        fill(NaN, 8),
        [0.013269520787476952, 0.04904223477325673, 0.0972163348468873, 0.1869715729381251, 0.24911587454854578, 0.24999282978133047, 0.24999186178779248, 0.249998867665936],
        165e-6, 780.0, [0, 0, -45e-6], 22e-3, 22e-3 / 780 .* (-0.5, 0.5)), # 03.22.2024 email
        "Alex 156" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.00458950737813959, 0.0452871875813540, 0.0825031976314792, 0.168777820405667, 0.247092954003264, 0.245070222870646, 0.239589096655836, 0.245928172893986],
        [0.0144389365963798, 0.0257698844742081, 0.0235326310441023, 0.0101418650354972, 0.00766915710252142, 0.0191547557185819, 0.0173949995998190, 0.00929358285744655],
        [1.4214278608874266e-6, 0.04459974029350741, 0.08982136711936144, 0.1769329320693605, 0.2480509143104852, 0.24997221921346477, 0.24997380448257234, 0.2499961141807692],
        156e-6, 760.0, [0, 0, -55e-6], 22e-3, 22e-3 / 760 .* (-0.5, 0.5)), # 09.10.2024 email
        "Alex 396 SP" => (-1 * [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.000930336702654036, 0.0219970375692036, 0.0495701070038793, 0.145412898483846, 0.220124943310094, 0.212762617024879, 0.229497023719215],
        [0.0136516127449087, 0.00926792988037959, 0.0183731521586386, 0.0154616957819848, 0.0192952858345390, 0.00904109790673457, 0.0166532760476917],
        [0.0026476940166382376, 0.022542676565879896, 0.056415926710378546, 0.15014517170414698, 0.22635141305411258, 0.2389042221175937, 0.2468757473482145],
        396e-6, 760.0, [0, 0, -55e-6], 22e-3, 22e-3 / 760 .* (-0.5, 0.5)), # 09.12.2024 email
        "Alex 396 DP" => (-1 * [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.00625835133319987, 0.0389417396508062, 0.0657703528331325, 0.149801707260384, 0.201686117678735, 0.228789128890909, 0.232158245203839],
        [0.0164622978125874, 0.0368946898175571, 0.0288861109867701, 0.0200166154396547, 0.00764205743802314, 0.00799219066141442, 0.0175996017865509],
        [0.0026476940166382376, 0.022542676565879896, 0.056415926710378546, 0.15014517170414698, 0.22635141305411258, 0.2389042221175937, 0.2468757473482145],
        396e-6, 760.0, [0, 0, -55e-6], 22e-3, 22e-3 / 760 .* (-0.5, 0.5)), # 09.12.2024 email
        "Alex 105" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.0373, 0.0503, 0.1181, 0.1929, 0.2597, 0.2289, 0.2680, 0.2405],
        fill(NaN, 8),
        [9.517667027831155e-8, 0.0628326911313974, 0.12222303371936649, 0.2126522625551978, 0.24995845400482292, 0.24999945328704146, 0.2499999166461335, 0.24999998829882425],
        105e-6, 800.0, [0, 0, -42e-6], 22e-3, 22e-3 / 800 .* (-0.5, 0.5)) # 08.21.2024 email
    )
    if haskey(experiments, name)
        params = experiments[name]
        return Experiment(name, params...)
    else
        throw(ArgumentError("No predefined experiment with the name \"$name\" found."))
    end
end

"""
    struct Simulation

A `Simulation` represents a particular simulation setup.

# Fields
- `type::String`: The type of simulation. Choose from `BE` and `WM`.
- `atom_number::Int64`: The number of atoms.
- `magnetic_field_computation_method::String`: The method used for calculating the magnetic field due to the wire. Choose from `"quadrupole"` and `"exact"`.
- `initial_μₑ::Union{String, Float64}`: The initial condition for the electron magnetic moments. Choose from `"up"`, `"down"`, `1/2`, or `-1/2`.
- `initial_μₙ::Union{String, Vector{Float64}}`: The sampling method for nuleus magnetic moments. Choose from `"HS"`, `"HS 2"`, `"HS 4"`, `"IHS"`, `"IHS 2"`, `"IHS 4"`, `"Iso"`, `"Iso 2"`, `"Iso 4"`, or give a vector of probability weights. The vector should be of length 2 or 4.
- `solver::String`: The differential equation solver. Several good ones are `"radau"`, `"radau5"`, `"RadauIIA5"`, and `"TRBDF2"`.
- `θₙ_is_fixed::Bool`: Whether `θₙ` is fixed or not.
- `branching_condition::String`: Which branching condition to use. Choose from `"B₀ dominant"` and `"Bₑ dominant"`.
- `BₙBₑ_strength::String`: The values for `Bₙ` and `Bₑ`. Choose from `"CQD"` and `"quantum"`.
- `Bₙ_ratio::Real`: The ratio of the used Bₙ to the theory value.
- `kᵢ::Float64`: The collapse coefficient.
- `average_method::Union{String, Tuple{String, Float64}}`: The average method. Choose from `"ABC"` (average angles then branching condition), `"BCA"` (branching conditions averaged), and `"no average"`. If using `"ABC"` or `"BCA"`, the input should be a tuple of length 2, where the second entry is the fraction of total time to be averaged. For example, `average_method = ("ABC", 1/16)`.
- `θ_cross_detection::Union{String, Tuple{String, Float64, Float64}, Tuple{String, Float64, String}}`: Whether angle cross is automatically detected. Choose from `"off"`, `"sign"`, and `"minabs"`. If using "sign" or "minabs", the input should be a tuple of length 3, where the second entry is the start time for detection, and the third entry is the period for detection. The period may be a fixed value or `"adaptive"`. For example, `θ_cross_detection = ("sign", 10e-6, 4.5e-6)`.
- `sigmoid_field::Union{String, Tuple{Float64, Float64}}`: Whether use a sigmoid transition field. Give either `"off"` or a tuple of length 2, where the first entry is the magnetic field strength, and the second entry is the y coordinate of the SG apparatus. For example, `sigmoid_field = (0.1, 2e-2)`.
- `R2_comparison::String`: How to calculate R2. Choose from `"experiment"` and `"qm"`.
"""
struct Simulation
    type::String
    atom_number::Int64
    magnetic_field_computation_method::String
    initial_μₑ::Union{String, Float64}
    initial_μₙ::Union{String, Vector{Float64}}
    solver::String
    θₙ_is_fixed::Bool
    branching_condition::String
    BₙBₑ_strength::String
    Bₙ_ratio::Real
    kᵢ::Float64
    average_method::Union{String, Tuple{String, Float64}}
    θ_cross_detection::Union{String, Tuple{String, Float64, Float64}, Tuple{String, Float64, String}}
    sigmoid_field::Union{String, Tuple{Float64, Float64}}
    R2_comparison::String
    function Simulation(
        type::String,
        atom_number::Int64,
        magnetic_field_computation_method::String,
        initial_μₑ::Union{String, Float64},
        initial_μₙ::Union{String, Vector{Float64}},
        solver::String,
        θₙ_is_fixed::Bool,
        branching_condition::String,
        BₙBₑ_strength::String,
        Bₙ_ratio::Real,
        kᵢ::Float64,
        average_method::Union{String, Tuple{String, Float64}},
        θ_cross_detection::Union{String, Tuple{String, Float64, Float64}, Tuple{String, Float64, String}},
        sigmoid_field::Union{String, Tuple{Float64, Float64}},
        R2_comparison::String
    )
        type ∈ ("BE", "WM") || throw(ArgumentError("The simulation type must be either BE or WM."))
        atom_number >= 1 || throw(ArgumentError("The number of atoms must be positive."))
        magnetic_field_computation_method ∈ ("quadrupole", "exact") || throw(ArgumentError("The magnetic field computation method must be either quadrupole or exact."))
        initial_μₑ ∈ ("up", "down", -1/2, 1/2) || throw(ArgumentError("The initial μₑ must be either up or down for BE type, or -1/2 or 1/2 for WM type."))
        (initial_μₙ isa String && initial_μₙ ∈ ("HS", "HS 2", "HS 4", "IHS", "IHS 2", "IHS 4", "Iso", "Iso 2", "Iso 4") && (initial_μₙ ∉ ("HS", "IHS", "Iso") || type == "BE")) || (initial_μₙ isa Vector{Float64} && (length(initial_μₙ) == 2 || length(initial_μₙ) == 4) && sum(initial_μₙ) ≈ 1 && all(x -> 0 <= x <= 1, initial_μₙ)) || throw(ArgumentError("The initial μₙ is invalid. See help of `Simulation`."))
        branching_condition ∈ ("B₀ dominant", "Bₑ dominant") || throw(ArgumentError("The branching condition must be either B₀ dominant or Bₑ dominant."))
        BₙBₑ_strength ∈ ("CQD", "quantum") || throw(ArgumentError("The BₙBₑ strength must be either CQD or quantum."))
        Bₙ_ratio >= 0 || throw(ArgumentError("The Bₙ ratio must be positive."))
        kᵢ >= 0 || throw(ArgumentError("kᵢ must be nonnegative."))
        (average_method isa String && average_method == "off") || (average_method isa Tuple{String, Float64} && average_method[1] ∈ ("ABC", "BCA") && 0 < average_method[2] < 1) || throw(ArgumentError("The average method is invalid. See help of `Simulation`."))
        (θ_cross_detection isa String && θ_cross_detection == "off") || (θ_cross_detection isa Tuple && θ_cross_detection[1] ∈ ("sign", "minabs") && θ_cross_detection[2] > 0 && ((θ_cross_detection[3] isa Float64 && θ_cross_detection[3] > 0) || (θ_cross_detection[3] isa String && θ_cross_detection[3] == "adaptive"))) || throw(ArgumentError("The θ cross detection is invalid. See help of `Simulation`."))
        (sigmoid_field isa String && sigmoid_field == "off") || (sigmoid_field isa Tuple{Float64, Float64} && sigmoid_field[2] >= 0) || throw(ArgumentError("The sigmoid field is invalid. See help of `Simulation`."))
        R2_comparison ∈ ("experiment", "qm") || throw(ArgumentError("The R2 comparison must be either experiment or qm."))
        if initial_μₙ isa Vector{Float64}
            @warn "The initial μₙ ia a vector. The consistency of the initial conditions needs manual check."
        end
        inconsistent_combinations = (Set([
            ("up", "IHS"),
            (-1/2, "IHS"),
            ("down", "HS"),
            (1/2, "HS"),
        ]), Set([
            ("up", "HS"),
            (-1/2, "HS"),
            ("down", "IHS"),
            (1/2, "IHS"),
        ]))
        if ((!θₙ_is_fixed || magnetic_field_computation_method == "exact") && (initial_μₑ, split(initial_μₙ, " ")) ∈ inconsistent_combinations[1]) || (θₙ_is_fixed && magnetic_field_computation_method == "quadrupole" && (initial_μₑ, split(initial_μₙ, " ")) ∈ inconsistent_combinations[2])
            @warn "The initial conditions are inconsistent with CQD postulates."
        end
        new(type, atom_number, magnetic_field_computation_method, initial_μₑ, initial_μₙ, solver, θₙ_is_fixed, branching_condition, BₙBₑ_strength, Bₙ_ratio, kᵢ, average_method, θ_cross_detection, sigmoid_field, R2_comparison)
    end
end

"""
    sample_atom_once(simulation::Simulation)

Sample one atom based on the simulation parameters.
"""
function sample_atom_once(simulation::Simulation)
    Bₙ, Bₑ = (simulation.BₙBₑ_strength == "CQD") ? (1.1884177310293015e-5, 0.05580626719338844) : (12.36e-3, 58.12)
    Bₙ *= simulation.Bₙ_ratio
    θₑ₀ = (simulation.initial_μₑ == "up" || simulation.initial_μₑ == -1/2) ? δθ / 2 : π - δθ / 2
    ϕₑ₀ = 0.0
    mᵢ = NaN
    if simulation.initial_μₙ == "HS"
        θₙ₀, ϕₙ₀ = 2asin(rand()^(1/4)), 2π * rand()
    elseif simulation.initial_μₙ == "IHS"
        θₙ₀, ϕₙ₀ = π - 2asin(rand()^(1/4)), 2π * rand()
    elseif simulation.initial_μₙ == "Iso"
        θₙ₀, ϕₙ₀ = 2asin(sqrt(rand())), 2π * rand()
    else
        if simulation.initial_μₙ isa Vector{Float64}
            mᵢs = length(simulation.initial_μₙ) == 2 ? [-3/2, 3/2] : [-3/2, -1/2, 1/2, 3/2]
            weights = simulation.initial_μₙ
        else
            dists = Dict("HS 2" => ([-3/2, 3/2], [2/3, 1/3]), "IHS 2" => ([-3/2, 3/2], [1/3, 2/3]), "Iso 2" => ([-3/2, 3/2], [0.5, 0.5]), "HS 4" => ([-3/2, -1/2, 1/2, 3/2], [0.4, 0.3, 0.2, 0.1]), "IHS 4" => ([-3/2, -1/2, 1/2, 3/2], [0.1, 0.2, 0.3, 0.4]), "Iso 4" => ([-3/2, -1/2, 1/2, 3/2], [0.25, 0.25, 0.25, 0.25]))
            mᵢs = dists[simulation.initial_μₙ][1]
            weights = dists[simulation.initial_μₙ][2]
        end
        mᵢ = sample(mᵢs, ProbabilityWeights(weights))
        Bₙ /= abs(mᵢ) == 1/2 ? 3 : 1
        θₙ₀, ϕₙ₀ = (mᵢ > 0 ? δθ / 2 : π - δθ / 2, 0.0)
    end
    return [θₑ₀, θₙ₀, ϕₑ₀, ϕₙ₀, Bₙ, Bₑ, mᵢ]
end

"""
    sample_atoms(simulation::Simulation)

Sample atoms according to the predefined number of atoms.
"""
function sample_atoms(simulation::Simulation)
    return [sample_atom_once(simulation) for _ ∈ 1:simulation.atom_number]
end

"""
    transform_vectors(vectors::Union{Vector{Vector{<:Real}}, Vector{<:Real}})

Transform a vector (length M) of vectors (length N) to a vector of length N with elements of length M.

If the input is a vector but not a vector of vectors, return the vector itself.
"""
function transform_vectors(vectors::Union{Vector{<:Vector{<:Real}}, Vector{<:Real}})
    return vectors isa Vector{<:Real} ? vectors : [collect(v) for v ∈ zip(vectors...)]
end

"""
    get_external_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, simulation::Simulation)

Calculate the external magnetic field components `Bx`, `By`, `Bz` at a given time `t` and return a tuple.

# Notes
- +y is right, +z is up, +x is out of page.
- Due to a different definition of the sign of current, the expressions have an extra minus sign.
"""
function get_external_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, simulation::Simulation)
    y = t * experiment.v
    Bx = experiment.Bᵣ[1]
    if simulation.magnetic_field_computation_method == "quadrupole"
        current_factor = -current * μ₀ / (2π)
        inv_Bᵣ_sq_sum = 1 / (experiment.Bᵣ[2]^2 + experiment.Bᵣ[3]^2)
        y_NP = current_factor * experiment.Bᵣ[3] * inv_Bᵣ_sq_sum
        z_NP = -experiment.zₐ - current_factor * experiment.Bᵣ[2] * inv_Bᵣ_sq_sum
        G = 2π / (μ₀ * (-current))
        Bᵣ2_Bᵣ3 = 2 * experiment.Bᵣ[2] * experiment.Bᵣ[3]
        Bᵣ3_sq_minus_Bᵣ2_sq = experiment.Bᵣ[3]^2 - experiment.Bᵣ[2]^2
        By = G * (Bᵣ2_Bᵣ3 * (y - y_NP) - Bᵣ3_sq_minus_Bᵣ2_sq * z_NP)
        Bz = G * (Bᵣ2_Bᵣ3 * z_NP + Bᵣ3_sq_minus_Bᵣ2_sq * (y - y_NP))
    else
        G = μ₀ * (-current) / (2π * (experiment.zₐ^2 + y^2))
        By = G * experiment.zₐ + experiment.Bᵣ[2]
        Bz = -G * y + experiment.Bᵣ[3]
    end
    Bz += simulation.sigmoid_field == "off" ? 0.0 : simulation.sigmoid_field[1] * (1 / (1 + exp(-(y - simulation.sigmoid_field[2]) * 1e3)) + 1 / (1 + exp((y + simulation.sigmoid_field[2]) * 1e3)))
    return Bx, By, Bz
end

"""
    get_external_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, magnetic_field_computation_method::String, sigmoid_field::Union{String, Tuple{Float64, Float64}})

Calculate the external magnetic field components `Bx`, `By`, `Bz` at a given time `t` and return a tuple.

# Notes
- +y is right, +z is up, +x is out of page.
- Due to a different definition of the sign of current, the expressions have an extra minus sign.
"""
function get_external_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, magnetic_field_computation_method::String, sigmoid_field::Union{String, Tuple{Float64, Float64}})
    y = t * experiment.v
    Bx = experiment.Bᵣ[1]
    if magnetic_field_computation_method == "quadrupole"
        current_factor = -current * μ₀ / (2π)
        inv_Bᵣ_sq_sum = 1 / (experiment.Bᵣ[2]^2 + experiment.Bᵣ[3]^2)
        y_NP = current_factor * experiment.Bᵣ[3] * inv_Bᵣ_sq_sum
        z_NP = -experiment.zₐ - current_factor * experiment.Bᵣ[2] * inv_Bᵣ_sq_sum
        G = 2π / (μ₀ * (-current))
        Bᵣ2_Bᵣ3 = 2 * experiment.Bᵣ[2] * experiment.Bᵣ[3]
        Bᵣ3_sq_minus_Bᵣ2_sq = experiment.Bᵣ[3]^2 - experiment.Bᵣ[2]^2
        By = G * (Bᵣ2_Bᵣ3 * (y - y_NP) - Bᵣ3_sq_minus_Bᵣ2_sq * z_NP)
        Bz = G * (Bᵣ2_Bᵣ3 * z_NP + Bᵣ3_sq_minus_Bᵣ2_sq * (y - y_NP))
    else
        G = μ₀ * (-current) / (2π * (experiment.zₐ^2 + y^2))
        By = G * experiment.zₐ + experiment.Bᵣ[2]
        Bz = -G * y + experiment.Bᵣ[3]
    end
    Bz += sigmoid_field == "off" ? 0.0 : sigmoid_field[1] * (1 / (1 + exp(-(y - sigmoid_field[2]) * 1e3)) + 1 / (1 + exp((y + sigmoid_field[2]) * 1e3)))
    return Bx, By, Bz
end

"""
    get_external_magnetic_fields_at_ends(current::Float64, experiment::Experiment, simulation::Simulation)

Calculate the external magnetic field components at the two ends of the flipper.
"""
function get_external_magnetic_fields_at_ends(current::Float64, experiment::Experiment, simulation::Simulation)
    (t₁, t₂) = experiment.time_span
    return (get_external_magnetic_fields(t₁, current, experiment, simulation), get_external_magnetic_fields(t₂, current, experiment, simulation))
end

"""
    spherical_to_cartesian(r::Real, θ::Real, ϕ::Real)

Convert spherical coordinates (r, θ, ϕ) to Cartesian coordinates (x, y, z).
"""
function spherical_to_cartesian(r::Real, θ::Real, ϕ::Real)
    r > 0 || throw(ArgumentError("The radius must be positive."))
    0 <= θ <= π || throw(ArgumentError("The θ angle must be between 0 and π."))
    x = r * sin(θ) * cos(ϕ)
    y = r * sin(θ) * sin(ϕ)
    z = r * cos(θ)
    return x, y, z
end

"""
    cartesian_to_spherical(x::Real, y::Real, z::Real)

Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, ϕ).
"""
function cartesian_to_spherical(x::Real, y::Real, z::Real)
    r = sqrt(x^2 + y^2 + z^2)
    r > 0 || throw(ArgumentError("Cannot convert the origin."))
    θ = acos(z / r)
    ϕ = atan(y, x)
    return r, θ, ϕ
end

"""
    get_perpendicular_norm_vector(v::Vector{<:Real})

Find one normalized vector that is perpendicular to v.
"""
function get_perpendicular_norm_vector(v::Vector{<:Real})
    length(v) == 3 || throw(ArgumentError("The vector must be 3D."))
    !(norm(v) ≈ 0) || throw(ArgumentError("The vector must be non-zero."))
    perpendicular_vector = [1, 0, 0] × v
    if norm(perpendicular_vector) ≈ 0
        perpendicular_vector = [0, 1, 0] × v
    end
    return perpendicular_vector / norm(perpendicular_vector)
end

"""
    transform_angles(θ::Union{Real, Vector{<:Real}}, ϕ::Union{Real, Vector{<:Real}}, z₁::Vector{<:Real}, z₂::Vector{<:Real})

Express θ and ϕ, which are defined in the z₁ spherical coordinates, in the z₂ spherical coordinates.

# Notes
- There are infinitly many possible rotations. Here only the simpliest case is considered.
"""
function transform_angles(θ::Union{Real, Vector{<:Real}}, ϕ::Union{Real, Vector{<:Real}}, z₁::Vector{<:Real}, z₂::Vector{<:Real})
    length(θ) == length(ϕ) || throw(ArgumentError("θ and ϕ must be of the same length."))
    (!(norm(z₁) ≈ 0) && !(norm(z₂) ≈ 0)) || throw(ArgumentError("The z vectors cannot be zero."))
    rotation_angle = acos(z₁ ⋅ z₂ / (norm(z₁) * norm(z₂)))
    rotation_axis = norm(z₁ × z₂) ≈ 0 ? get_perpendicular_norm_vector(z₁) : z₁ × z₂ / norm(z₁ × z₂)
    rotation_matrix = AngleAxis(-rotation_angle, rotation_axis...)
    cartesian_coordinates_z₁ = collect.(spherical_to_cartesian.(1, θ, ϕ))
    cartesian_coordinates_z₂ = length(θ) == 1 ? collect(rotation_matrix * cartesian_coordinates_z₁) : collect.(Ref(rotation_matrix) .* cartesian_coordinates_z₁)
    (_, θₚ, ϕₚ) = length(θ) == 1 ? cartesian_to_spherical(cartesian_coordinates_z₂...) : transform_vectors(collect.(cartesian_to_spherical.(transform_vectors(cartesian_coordinates_z₂)...)))
    return θₚ, ϕₚ
end

"""
    apply_branching(θₑs::Union{Real, Vector{<:Real}}, θₙs::Union{Real, Vector{<:Real}}, ϕₑs::Union{Real, Vector{<:Real}}, ϕₙs::Union{Real, Vector{<:Real}}, mᵢ::Float64, simulation::Simulation)

Apply the branching conditions.
"""
function apply_branching(θₑs::Union{Real, Vector{<:Real}}, θₙs::Union{Real, Vector{<:Real}}, ϕₑs::Union{Real, Vector{<:Real}}, ϕₙs::Union{Real, Vector{<:Real}}, mᵢ::Float64, simulation::Simulation)
    all(x -> 0 <= x <= π, θₑs) || throw(ArgumentError("θₑs must be between 0 and π."))
    all(x -> 0 <= x <= π, θₙs) || throw(ArgumentError("θₙs must be between 0 and π."))
    initial_μₑ_is_up = simulation.initial_μₑ == "up" || simulation.initial_μₑ == -1/2
    if simulation.type == "BE"
        if simulation.branching_condition == "B₀ dominant"
            return initial_μₑ_is_up ? (θₑs .> θₙs) : (θₑs .< θₙs)
        else
            μₑ_unit_vectors = transform_vectors([sin.(θₑs) .* cos.(ϕₑs), sin.(θₑs) .* sin.(ϕₑs), cos.(θₑs)])
            μₙ_unit_vectors = transform_vectors([sin.(θₙs) .* cos.(ϕₙs), sin.(θₙs) .* sin.(ϕₙs), cos.(θₙs)])
            B_unit_vector = [0, 0, 1]
            θₑBs = μₑ_unit_vectors isa Vector{<:Vector{<:Real}} ? acos.(μₑ_unit_vectors .⋅ Ref(B_unit_vector)) : acos(μₑ_unit_vectors ⋅ B_unit_vector)
            θₑₙs = μₑ_unit_vectors isa Vector{<:Vector{<:Real}} ? acos.(μₑ_unit_vectors .⋅ μₙ_unit_vectors) : acos(μₑ_unit_vectors ⋅ μₙ_unit_vectors)
            return initial_μₑ_is_up ? (θₑBs .> θₑₙs) : (θₑBs .< θₑₙs)
        end
    else
        j, mᵢi, pool = (simulation.initial_μₙ ∈ ("HS 2", "IHS 2", "Iso 2") || (simulation.initial_μₙ isa Vector{Float64}) && length(simulation.initial_μₙ) == 2) ? (1/2, mᵢ / 3, [-3/2, 3/2]) : (3/2, mᵢ, [-3/2, -1/2, 1/2, 3/2])
        CCQ_weights = [[WignerD.wignerdjmn(j, mᵢi, k, θₙs[l])^2 for k ∈ -j:j] for l ∈ eachindex(θₙs)]
        mᵢfs = [sample(pool, ProbabilityWeights(CCQ_weights[l])) for l ∈ eachindex(CCQ_weights)]
        if length(mᵢfs) == 1
            mᵢfs = mᵢfs[1]
        end
        if simulation.branching_condition == "B₀ dominant"
            return initial_μₑ_is_up ? mᵢfs .> 0 : mᵢfs .< 0
        else
            θₙsₚ = ifelse.(mᵢfs .> 0, δθ, π - δθ)
            μₑ_unit_vectors = transform_vectors([sin.(θₑs) .* cos.(ϕₑs), sin.(θₑs) .* sin.(ϕₑs), cos.(θₑs)])
            μₙ_unit_vectors = transform_vectors([sin.(θₙsₚ) .* cos.(0), sin.(θₙsₚ) .* sin.(0), cos.(θₙsₚ)])
            B_unit_vector = [0, 0, 1]
            θₑBs = μₑ_unit_vectors isa Vector{<:Vector{<:Real}} ? acos.(μₑ_unit_vectors .⋅ Ref(B_unit_vector)) : acos(μₑ_unit_vectors ⋅ B_unit_vector)
            θₑₙs = μₑ_unit_vectors isa Vector{<:Vector{<:Real}} ? acos.(μₑ_unit_vectors .⋅ μₙ_unit_vectors) : acos(μₑ_unit_vectors ⋅ μₙ_unit_vectors)
            is_equal = θₑBs .≈ θₑₙs
            flips_for_up = θₑBs .> θₑₙs
            if is_equal isa Bool
                if is_equal
                    flips_for_up = sample([true, false], ProbabilityWeights([sin(θₑBs / 2)^2, cos(θₑBs / 2)^2]))
                end
            else
                flips_for_up[is_equal] .= [sample([true, false], ProbabilityWeights([sin(i / 2)^2, cos(i / 2)^2])) for i ∈ θₑBs[is_equal]]
            end
            return initial_μₑ_is_up ? flips_for_up : .!flips_for_up
        end
    end
end

"""
    is_flipped(angles::Union{Vector{Float64}, Vector{Vector{Float64}}}, mᵢ::Float64, current::Float64, experiment::Experiment, simulation::Simulation)

Determine whether an atom has flipped based on its angles, magnetic field computation method, and the branching condition.
"""
function is_flipped(angles::Union{Vector{Float64}, Vector{Vector{Float64}}}, mᵢ::Float64, current::Float64, experiment::Experiment, simulation::Simulation)
    θₑfs, θₙfs, ϕₑfs, ϕₙfs = transform_vectors(angles)
    (B_start, B_end) = get_external_magnetic_fields_at_ends(current, experiment, simulation)
    B_unit_vectors_at_ends = collect.((B_start ./ norm(B_start), B_end ./ norm(B_end)))
    θₑfsf, ϕₑfsf = transform_angles(θₑfs, ϕₑfs, B_unit_vectors_at_ends[1], B_unit_vectors_at_ends[2])
    θₙfsf, ϕₙfsf = transform_angles(θₙfs, ϕₙfs, B_unit_vectors_at_ends[1], B_unit_vectors_at_ends[2])
    return apply_branching(θₑfsf, θₙfsf, ϕₑfsf, ϕₙfsf, mᵢ, simulation)
end

"""
    wrap(θ)

Wrap the angle θ to between 0 and π.
"""
function wrap(θ)
    return π .- abs.(mod.(θ, 2π) .- π)
end

"""
    CQD_LLG_equation!(du, u, p, t)

Define the differential equation using CQD LLG equations.

# Arguments
`u = [θₑ, θₙ, ϕₑ, ϕₙ]`: The variable of the differential equation.
`p = [experiment::Experiment, simulation::Simulation, current, Bₙ, Bₑ]`: The parameters passed to the solver.
"""
function CQD_LLG_equation!(du, u, p, t)
    experiment, simulation, current, Bₙ, Bₑ = p
    θₑ, θₙ, ϕₑ, ϕₙ = u
    Bx, By, Bz = get_external_magnetic_fields(t, current, experiment, simulation)
    sin_θₑ, cos_θₑ = sincos(θₑ)
    sin_θₙ, cos_θₙ = sincos(θₙ)
    sin_ϕₑ, cos_ϕₑ = sincos(ϕₑ)
    sin_ϕₙ, cos_ϕₙ = sincos(ϕₙ)
    Bₙ_sin_θₙ = Bₙ * sin_θₙ
    Bₑ_sin_θₑ = Bₑ * sin_θₑ
    Δϕₑₙ = ϕₑ - ϕₙ
    Δϕₙₑ = -Δϕₑₙ
    θₑ_wrapped = wrap(θₑ)
    θₙ_wrapped = wrap(θₙ)
    du₁ = -γₑ * (By * cos_ϕₑ - Bx * sin_ϕₑ + Bₙ_sin_θₙ * sin(Δϕₙₑ))
    du₂ = -γₙ * (By * cos_ϕₙ - Bx * sin_ϕₙ + Bₑ_sin_θₑ * sin(Δϕₑₙ))
    du₃, dϕₑ = 0.0, 0.0
    du₄, dϕₙ = 0.0, 0.0
    if θₑ_wrapped >= δθ && π - θₑ_wrapped >= δθ
        cot_θₑ = cos_θₑ / sin_θₑ
        du₃ = -γₑ * (Bz + Bₙ * cos_θₙ - cot_θₑ * (Bx * cos_ϕₑ + By * sin_ϕₑ + Bₙ_sin_θₙ * cos(Δϕₙₑ)))
        dϕₑ = du₃ - sign(du₃) * simulation.kᵢ * abs(du₁) * csc(θₑ)
    end
    if θₙ_wrapped >= δθ && π - θₙ_wrapped >= δθ
        cot_θₙ = cos_θₙ / sin_θₙ
        du₄ = -γₙ * (Bz + Bₑ * cos_θₑ - cot_θₙ * (Bx * cos_ϕₙ + By * sin_ϕₙ + Bₑ_sin_θₑ * cos(Δϕₑₙ)))
        dϕₙ = du₄ - sign(du₄) * simulation.kᵢ * abs(du₂) * csc(θₙ)
    end
    dθₑ = du₁ - sign(θₙ - θₑ) * simulation.kᵢ * abs(du₃) * sin_θₑ
    dθₙ = du₂ - sign(θₑ - θₙ) * simulation.kᵢ * abs(du₄) * sin_θₙ
    if simulation.θₙ_is_fixed
        dθₙ = 0.0
    end
    du[1], du[2], du[3], du[4] = dθₑ, dθₙ, dϕₑ, dϕₙ
end

"""
    latex_exponential(x::Real)

Convert a number `x` to a beautiful scientific-notation latex string.
"""
function latex_exponential(x::Real)
    if x == 0
        return "0"
    else
        exponent = floor(Int, log10(abs(x)))
        mantissa = x / 10.0^exponent
        return mantissa == 1 ? "10^{$exponent}" : "$mantissa\\times10{$exponent}"
    end
end

"""
    get_solver(name::String)

Get the solver corresponding to the name.
"""
function get_solver(name::String)
    try
        return @eval $(Symbol(name))()
    catch
        throw(ArgumentError("Unknown solver: $name."))
    end
end

"""
    simulate(experiment::Experiment, simulation::Simulation)

Simulate the whole system.
"""
function simulate(experiment::Experiment, simulation::Simulation)
    if simulation.magnetic_field_computation_method == "quadrupole" && experiment.name ∈ ("FS Low \$z_a\$", "FS High \$z_a\$")
        error("Quadrupole is calculated for zero current. Simulation will be slow and unreliable.")
    end
    raw_data = falses(length(experiment.currents), simulation.atom_number)
    θₑ_plot, θₙ_plot, θₑθₙ_plot = plot(), plot(), plot()
    for i ∈ eachindex(experiment.currents)
        current_i = experiment.currents[i]
        print("current=$i: ")
        atoms = sample_atoms(simulation)
        mᵢs = transform_vectors(atoms)[end]
        u₀ = atoms[1][1:4]
        ode_prob = ODEProblem(CQD_LLG_equation!, u₀, experiment.time_span, (experiment, simulation, current_i, atoms[1][5], atoms[1][6]))
        ensemble_prob = EnsembleProblem(ode_prob, prob_func = (prob, i, repeat) -> remake(prob, u0 = atoms[i][1:4], p = (experiment, simulation, current_i, atoms[i][5], atoms[i][6])))
        @time solution = solve(ensemble_prob, get_solver(simulation.solver), EnsembleDistributed(), trajectories=simulation.atom_number, reltol=1e-6, abstol=1e-6, dtmin=1e-30, force_dtmin=true, maxiters=1e14, saveat=2e-8, dt=1e-30)
        for j ∈ 1:simulation.atom_number
            sol = solution[j]
            ϕₑf, ϕₙf = sol.u[end - 1][3], sol.u[end - 1][4]
            mᵢ = mᵢs[j]
            step_number = length(sol.t)
            if simulation.average_method == "off"
                θₑf, θₙf, ϕₑf, ϕₙf = wrap(sol.u[end - 1][1]), wrap(sol.u[end - 1][2]), sol.u[end - 1][3], sol.u[end - 1][4]
                raw_data[i, j] = is_flipped([θₑf, θₙf, ϕₑf, ϕₙf], mᵢ, current_i, experiment, simulation)
            else
                start_index = step_number - trunc(Int, step_number * simulation.average_method[2])
                average_index_range = start_index:(step_number - 1)
                if simulation.average_method[1] == "ABC" # average angles then apply branching condition
                    θₑfs, θₙfs, ϕₑfs, ϕₙfs = wrap([sol.u[k][1] for k ∈ average_index_range]), wrap([sol.u[k][2] for k ∈ average_index_range]), [sol.u[k][3] for k ∈ average_index_range], [sol.u[k][4] for k ∈ average_index_range]
                    raw_data[i, j] = is_flipped([mean(θₑfs), mean(θₙfs), mean(ϕₑfs), mean(ϕₙfs)], mᵢ, current_i, experiment, simulation)
                else # apply branching condition then average
                    θₑfs, θₙfs, ϕₑfs, ϕₙfs = wrap([sol.u[k][1] for k ∈ average_index_range]), wrap([sol.u[k][2] for k ∈ average_index_range]), [sol.u[k][3] for k ∈ average_index_range], [sol.u[k][4] for k ∈ average_index_range]
                    angles = [collect(x) for x ∈ zip(θₑfs, θₙfs, ϕₑfs, ϕₙfs)]
                    flips = is_flipped(angles, mᵢ, current_i, experiment, simulation)
                    flip_mean = mean(flips)
                    raw_data[i, j] = flip_mean >= 0.5
                end
            end
        end
        sol = solution[1]
        step_number = length(sol.t)
        θₑfs, θₙfs = wrap([sol.u[k][1] for k ∈ 1:step_number]), wrap([sol.u[k][2] for k ∈ 1:step_number])
        plot!(θₑ_plot, sol.t * 1e6, θₑfs, label="\$$current_i\$ A", linestyle=:solid, dpi=300)
        xlabel!("Time \$t\$ [\$\\mu\$s]"); ylabel!("\$\\theta_e(t)\$"); title!("\$\\theta_e(t)\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); ylims!(0, π)
        plot!(θₙ_plot, sol.t * 1e6, θₙfs, label="\$$current_i\$ A", linestyle=:dash, dpi=300)
        xlabel!("Time \$t\$ [\$\\mu\$s]"); ylabel!("\$\\theta_n(t)\$"); title!("\$\\theta_n(t)\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); ylims!(0, π)
        plot!(θₑθₙ_plot, sol.t * 1e6, θₑfs, label="e, \$$current_i\$", linestyle=:solid, color=i)
        plot!(θₑθₙ_plot, sol.t * 1e6, θₙfs, label="n", linestyle=:dash, color=i, dpi=300)
        plot!(θₑθₙ_plot, legend=:best, legendcolumns=2)
        xlabel!("Time \$t\$ [\$\\mu\$s]"); ylabel!("Angles [rad]"); title!("Evolution of Angles, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); ylims!(0, π)
    end
    return raw_data, θₑ_plot, θₙ_plot, θₑθₙ_plot
end

"""
    struct Results

A `Results` represents the results of the simulation.

# Fields
- `raw_data::BitArray`: The raw simulation data of whether the atoms flip.
- `flip_probabilities::Vector{Float64}`: The flip probability calculated from the simulation.
- `flip_probabilities_stds::Vector{Float64}`: The standard deviations of the calculated flip probabilities.
- `R2::Float64`: The R sqaure value calculated from the simulation and experiment.
- `θₑ_plot::Plots.Plot`: The plot of θₑ dynamics.
- `θₙ_plot::Plots.Plot`: The plot of θₙ dynamics.
- `θₑθₙ_plot::Plots.Plot`: The plot that combines the two θ plots.
- `flip_plot::Plots.Plot`: The plot of the simulation and experiment flip probabilities.
"""
struct Results
    raw_data::BitArray
    flip_probabilities::Vector{Float64}
    flip_probabilities_stds::Vector{Float64}
    R2::Float64
    θₑ_plot::Plots.Plot
    θₙ_plot::Plots.Plot
    θₑθₙ_plot::Plots.Plot
    flip_plot::Plots.Plot
    function Results(raw_data::BitArray, flip_probabilities::Vector{Float64}, flip_probabilities_stds::Vector{Float64}, R2::Float64, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot, flip_plot::Plots.Plot)
        all(isnan, flip_probabilities_stds) || all(x -> x >= 0, flip_probabilities_stds) || throw(ArgumentError("The standard deviations of the flip probabilities are invalid."))
        new(raw_data, flip_probabilities, flip_probabilities_stds, R2, θₑ_plot, θₙ_plot, θₑθₙ_plot, flip_plot)
    end
end

"""
    Results(experiment::Experiment, simulation::Simulation, raw_data::BitArray, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot)

Outer constructor for `Results`.
"""
function Results(experiment::Experiment, simulation::Simulation, raw_data::BitArray, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot)
    function compute_flip_probabilities_R2_stds(raw_data, experiment_data)
        flip_number = dropdims(sum(raw_data, dims=2), dims=2)
        flip_probabilities = flip_number ./ size(raw_data, 2)
        sample_stds = dropdims(std(raw_data, dims=2), dims=2)
        flip_probabilities_stds = sample_stds / sqrt(size(raw_data, 2))
        term₁ = sum((experiment_data .- mean(experiment_data)).^2)
        term₂ = sum((experiment_data .- flip_probabilities).^2)
        R2 = 1 - term₂ / term₁
        return flip_probabilities, R2, flip_probabilities_stds
    end
    function plot_flip_probabilities(experiment::Experiment, simulation::Simulation, flip_probabilities, flip_probabilities_stds, R2)
        flip_plot = plot()
        plot_range = split(experiment.name, " ")[1] == "FS" ? (2:lastindex(experiment.currents)) : (1:lastindex(experiment.currents))
        scatter!(flip_plot, abs.(experiment.currents[plot_range]), experiment.flip_probabilities[plot_range], yerr = experiment.flip_probabilities_stds[plot_range], marker=(:xcross, 6), markerstrokewidth=3, markerstrokecolor=:auto, linewidth=2, label=experiment.name)
        scatter!(flip_plot, abs.(experiment.currents[plot_range]), experiment.qm_flip_probabilities[plot_range], marker=(:plus, 6), markerstrokewidth=3, label="QM Simulation")
        plot!(flip_plot, abs.(experiment.currents[plot_range]), flip_probabilities[plot_range], yerr = flip_probabilities_stds[plot_range], marker=(:circle, 4), label="CQD LLG Simulation", markerstrokecolor=:auto, legend=:best, dpi=300, minorgrid=true, xscale=:log10)
        xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("\$R^2=" * string(trunc(R2, digits=3)) * "\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); ylims!(0, 1)
        return flip_plot
    end
    flip_probabilities, R2, flip_probabilities_stds = compute_flip_probabilities_R2_stds(raw_data, simulation.R2_comparison == "experiment" ? experiment.flip_probabilities : experiment.qm_flip_probabilities)
    flip_plot = plot_flip_probabilities(experiment, simulation, flip_probabilities, flip_probabilities_stds, R2)
    return Results(raw_data, flip_probabilities, flip_probabilities_stds, R2, θₑ_plot, θₙ_plot, θₑθₙ_plot, flip_plot)
end

"""
    save_results(experiment::Experiment, simulation::Simulation, results::Results, start_time, file_dir)

Save the results of the whole simulation.
"""
function save_results(experiment::Experiment, simulation::Simulation, results::Results, start_time, file_dir)
    folder_dir = joinpath(file_dir, Dates.format(start_time, "yyyy-mm-dd_HH-MM-SS.sss"))
    isdir(folder_dir) || mkdir(folder_dir)
    df₁ = DataFrame(results.raw_data, :auto)
    CSV.write(joinpath(folder_dir, "Raw_Data.csv"), df₁)
    df₂ = DataFrame(results.flip_probabilities[:, :], :auto)
    CSV.write(joinpath(folder_dir, "Flip_Probabilities.csv"), df₂)
    savefig(results.θₑ_plot, joinpath(folder_dir, "θe_Plot.svg"))
    savefig(results.θₙ_plot, joinpath(folder_dir, "θn_Plot.svg"))
    savefig(results.θₑθₙ_plot, joinpath(folder_dir, "θeθn_Plot.svg"))
    savefig(results.flip_plot, joinpath(folder_dir, "Flip_Plot.svg"))
    cp(@__FILE__, joinpath(folder_dir, "BECQDBase.jl"))
    end_time = now()
    metadata = OrderedDict(
        "Experiment" => experiment.name,
        "Simulation Type" => simulation.type,
        "Number of Atoms" => simulation.atom_number,
        "Magnetic Field Computation Method" => simulation.magnetic_field_computation_method,
        "Initial μₑ [relative to local B_ext]" => simulation.initial_μₑ,
        "Initial μₙ [relative to local B_ext]" => simulation.initial_μₙ,
        "Differential Equations Solver" => simulation.solver,
        "θₙ" => simulation.θₙ_is_fixed ? "fixed" : "vary",
        "Branching Condition" => simulation.branching_condition,
        "Bₙ Bₑ Strength" => simulation.BₙBₑ_strength,
        "Bₙ Ratio" => simulation.Bₙ_ratio,
        "kᵢ" => simulation.kᵢ,
        "Average Method" => simulation.average_method,
        "θ Cross Detection" => simulation.θ_cross_detection,
        "Sigmoid Field" => simulation.sigmoid_field,
        "R2 Comparison" => simulation.R2_comparison,
        "Simulation Start Time" => Dates.format(start_time, "yyyy-mm-dd HH:MM:SS.sss"),
        "Simulation End Time" => Dates.format(end_time, "yyyy-mm-dd HH:MM:SS.sss"),
        "Simulation Run Time" => string(Dates.canonicalize(end_time - start_time)),
        "Remnant Fields [T]" => experiment.Bᵣ,
        "Atom Speed [m/s]" => experiment.v,
        "zₐ [m]" => experiment.zₐ,
        "Flight Path Length [mm]" => experiment.system_length * 1e3,
        "Flight Time Range [μs]" => experiment.time_span .* 1e6,
        "Wire Currents [A]" => experiment.currents,
        "Experiment Flip Probabilities" => experiment.flip_probabilities,
        "Experiment Flip Probabilities Standard Deviations" => replace(experiment.flip_probabilities_stds, NaN => nothing),
        "QM Flip Probabilities" => experiment.qm_flip_probabilities,
        "Simulation Flip Probabilities" => results.flip_probabilities,
        "Simulation Flip Probabilities Standard Deviations" => replace(results.flip_probabilities_stds, NaN => nothing),
        "R2" => results.R2,
        "δθ" => δθ,
        "Machine" => Sys.MACHINE,
        "CPU" => Sys.cpu_info()[1].model,
        "Total Memory [GB]" => Sys.total_memory() / 1024^3,
        "Julia Version" => VERSION
    )
    open(joinpath(folder_dir, "Info.json"), "w") do file
        JSON3.pretty(file, JSON3.write(metadata))
        println(file)
    end
    pkg_info = Pkg.dependencies()
    pkg_data = OrderedDict()
    for (uuid, info) in pkg_info
        pkg_data[info.name] = info.version
    end
    open(joinpath(folder_dir, "Pkg_Info.json"), "w") do file
        JSON3.pretty(file, JSON3.write(pkg_data))
        println(file)
    end
end

end