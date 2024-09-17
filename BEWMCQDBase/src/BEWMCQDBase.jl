#=
BEWMCQDBase.jl

This module defines important structures and functions for CQD Wigner d simulations.
Author: Xukun Lin
Date: 09/16/2024

Required packages: "Pkg", "Dates", "LinearAlgebra", "Statistics", "Logging", "StatsBase", "DifferentialEquations", "Plots", "DataStructures", "DataFrames", "CSV", "LaTeXStrings", "JSON3", "WignerD".
Required constants: μ₀, γₑ, γₙ.
=#

module BEWMCQDBase

using Pkg, Dates, LinearAlgebra, Statistics, Logging, StatsBase, DifferentialEquations, Plots, DataStructures, DataFrames, CSV, LaTeXStrings, JSON3, WignerD

export μ₀, γₑ, γₙ
export Experiment, Simulation, Result
export sample_atom_once, sample_atoms, is_flipped, get_magnetic_fields, simulate, save_results

const μ₀ = 4π * 1e-7
const γₑ = -1.76085963e11
const γₙ = 1.2500612e7

"""
    struct Experiment

An `Experiment` represents the experiment to simulate.

# Fields
- `name::String`: The name of the experiment. Predefined experiment names are `"04.06.2024 Alex"`, `"04.18.2024 Alex"`, `"04.23.2024 Alex"`, `"08.21.2024 Alex"`, `"FS Low \$z_a\$"`, and `"FS High \$z_a\$"`. For predefined experiments, use `Experiment(name)`.
- `current::Vector{Float64}`: The wire currents.
- `flip_probability::Vector{Float64}`: The flip probabilities from the experiment.
- `zₐ::Float64`: The distance from the beam to the null point.
- `v::Float64`: The velocity of the atomic beam.
- `Bᵣ::Vector{Float64}`: The remnant field.
- `system_length::Float64`: The length of the system.
- `time_span::Tuple{Float64, Float64}`: The flight time range for the atoms.
"""
struct Experiment
    name::String
    current::Vector{Float64}
    flip_probability::Vector{Float64}
    zₐ::Float64
    v::Float64
    Bᵣ::Vector{Float64}
    system_length::Float64
    time_span::Tuple{Float64, Float64}
    function Experiment(name::String, current::Vector{Float64}, flip_probability::Vector{Float64}, zₐ::Float64, v::Float64, Bᵣ::Vector{Float64}, system_length::Float64, time_span::Tuple{Float64, Float64})
        length(current) == length(flip_probability) || throw(ArgumentError("The length of the current list and the flip probability list must be equal."))
        all(0 .<= flip_probability .<= 1) || throw(ArgumentError("Flip probability must be between 0 and 1."))
        zₐ > 0 || throw(ArgumentError("zₐ must be positive."))
        v > 0 || throw(ArgumentError("v must be positive."))
        system_length > 0 || throw(ArgumentError("The length of the system must be positive."))
        new(name, current, flip_probability, zₐ, v, Bᵣ, system_length, time_span)
    end
end

"""
    Experiment(name::String)

Outer constructor for `Experiment` that takes a predefined experiment name.
"""
function Experiment(name::String)
    experiments = Dict(
        "FS Low \$z_a\$" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.19, 6.14, 14.87, 26.68, 30.81, 26.8, 12.62, 0.1] / 100,
        105e-6, 800.0, [0, 0, 42e-6], 16e-3, 16e-3 / 800 .* (-0.5, 0.5)),
        "FS High \$z_a\$" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [2.4, 5.77, 12.1, 15.28, 25.38, 25.63, 17.34, 0.19] / 100,
        225e-6, 800.0, [0, 0, 42e-6], 16e-3, 16e-3 / 800 .* (-0.5, 0.5)),
        "04.06.2024 Alex" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.0534151282288409, 0.0479013706009164, 0.103178291177038, 0.182199609641768, 0.271739445083921, 0.265583816803359, 0.251739022695542, 0.253387509472528],
        165e-6, 780.0, [0, 0, -45e-6], 22e-3, 22e-3 / 780 .* (-0.5, 0.5)),
        "04.18.2024 Alex" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.0128844158263087, 0.0506085714485169, 0.0871382583394999, 0.172233902905342, 0.246648689269622, 0.247096508139797, 0.240236993651768, 0.246302881099092],
        156e-6, 740.0, [0, 0, -54.8e-6], 22e-3, 22e-3 / 740 .* (-0.5, 0.5)),
        "04.23.2024 Alex" => (-1 * [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.00797624927333513, 0.0394177574956492, 0.0641029090865451, 0.146194830476275, 0.203354577568627, 0.224462526145968, 0.228983398875884],
        396e-6, 740.0, [0, 0, -55e-6], 22e-3, 22e-3 / 740 .* (-0.5, 0.5)),
        "08.21.2024 Alex" => (-1 * [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
        [0.0373, 0.0503, 0.1181, 0.1929, 0.2597, 0.2289, 0.2680, 0.2405],
        105e-6, 800.0, [0, 0, -55e-6], 22e-3, 22e-3 / 800 .* (-0.5, 0.5))
    )
    if haskey(experiments, name)
        params = experiments[name]
        return Experiment(name, params...)
    else
        throw(ArgumentError("No predefined experiment with the name '$name' found."))
    end
end

"""
    struct Simulation

A `Simulation` represents a particular simulation setup.

# Fields
- `atom_number::Int64`: The number of atoms.
- `magnetic_field_computation_method::String`: The method used for calculating the magnetic field due to the wire. Choose from `"quadrupole"` and `"exact"`.
- `initial_m_S::Float64`: The initial condition for the electron magnetic moments. Choose from `-1/2` and `1/2`.
- `initial_μₙ_sampling_weight::Vector{Float64}`: The sampling weight for nuleus magnetic moments.
- `g_ratio_index::Int64`: Choose the method to calculate g. Choose from `[1, 2, 3]`.
- `solver`: The differential equation solver. Several good ones are `radau()`, `radau5()`, `RadauIIA5()`, and `TRBDF2()`.
- `branching_condition::String`: (Optional) Which branching condition to use. Defaults to `"original"`. Choose from `"original"` and `"revised"`.
- `BₙBₑ_strength::String`: (Optional) The values for `Bₙ` and `Bₑ`. Defaults to `"CQD"`. Choose from `"CQD"` and `"quantum"`.
- `δθ::Float64`: (Optional) The small angle used for warning handling. Defaults to `1e-6`.
- `kᵢ::Float64`: (Optional) The collapse coefficient. Dafaults to `0.0`.
- `average_method::String`: (Optional) The average method. Defaults to `"none"`. Choose from `"angle then branching"`, `"branching then angle"`, and `"none"`.
- `θ_cross_is_detected::Bool`: (Optional) Whether angle cross is automatically detected. Defaults to `false`.
- `θ_cross_detection_method::String`: (Optional) The method used to detection θ cross. Defaults to `"sign"`. Choose from `"sign"` and `"minabs"`.
- `detection_start_time::Float64`: (Optional) When the detection starts. Defaults to `10e-6`.
- `detection_period::Union{Float64, String}`: (Optional) The time step size for detection. Defaults to `4.5e-6`. Choose from a `Float64` and `"adaptive"`.
- `B_SG::Float64`: (Optional) The magnetic field for the SG apparatus. Defaults to `0.0`.
- `y_SG::Float64`: (Optional) The y coordinate of the SG apparatus. Defaults to `2e-2`.
"""
struct Simulation
    atom_number::Int64
    magnetic_field_computation_method::String
    initial_m_S::Float64
    initial_μₙ_sampling_weight::Vector{Float64}
    g_ratio_index::Int64
    solver
    branching_condition::String
    BₙBₑ_strength::String
    δθ::Float64
    kᵢ::Float64
    average_method::String
    θ_cross_is_detected::Bool
    θ_cross_detection_method::String
    detection_start_time::Float64
    detection_period::Union{Float64, String}
    B_SG::Float64
    y_SG::Float64
    function Simulation(
        atom_number::Int64,
        magnetic_field_computation_method::String,
        initial_m_S::Float64,
        initial_μₙ_sampling_weight::Vector{Float64},
        g_ratio_index::Int64,
        solver;
        branching_condition::String="original",
        BₙBₑ_strength::String="CQD",
        δθ::Float64=1e-6,
        kᵢ::Float64=0.0,
        average_method::String="none",
        θ_cross_is_detected::Bool=false,
        θ_cross_detection_method::String="sign",
        detection_start_time::Float64=10e-6,
        detection_period::Union{Float64, String}=4.5e-6,
        B_SG::Float64=0.0,
        y_SG::Float64=2e-2
    )
        atom_number >= 1 || throw(ArgumentError("The number of atoms must be positive."))
        magnetic_field_computation_method ∈ ["quadrupole", "exact"] || throw(ArgumentError("The magnetic field computation method must be either quadrupole or exact."))
        initial_m_S ∈ [-1/2, 1/2] || throw(ArgumentError("The initial μₑ m_S must be either -1/2 or 1/2."))
        sum(initial_μₙ_sampling_weight) ≈ 1.0 || throw(ArgumentError("The initial μₙ sampling weight must add up to 1."))
        all(x -> x >= 0 && x <= 1, initial_μₙ_sampling_weight) || throw(ArgumentError("The initial μₙ sampling weight is out of bounds."))
        g_ratio_index ∈ [1, 2, 3] || throw(ArgumentError("The g ratio index must be 1, 2, or 3."))
        branching_condition ∈ ["original", "revised"] || throw(ArgumentError("The branching condition must be either original or revised."))
        BₙBₑ_strength ∈ ["CQD", "quantum"] || throw(ArgumentError("The BₙBₑ strength must be either CQD or quantum."))
        δθ > 0 || throw(ArgumentError("δθ must be positive."))
        kᵢ >= 0 || throw(ArgumentError("kᵢ must be nonnegative."))
        average_method ∈ ["none", "angle then branching", "branching then angle"] || throw(ArgumentError("The average method is invalid. See help of `Simulation`."))
        θ_cross_detection_method ∈ ["sign", "minabs"] || throw(ArgumentError("The θ cross detection method must be either sign or minabs"))
        detection_start_time > 0 || throw(ArgumentError("The detection start time must be positive."))
        (detection_period isa Float64 && detection_period > 0) || detection_period == "adaptive" || throw(ArgumentError("The detection period is invalid."))
        new(atom_number, magnetic_field_computation_method, initial_m_S, initial_μₙ_sampling_weight, g_ratio_index, solver, branching_condition, BₙBₑ_strength, δθ, kᵢ, average_method, θ_cross_is_detected, θ_cross_detection_method, detection_start_time, detection_period, B_SG, y_SG)
    end
end

function sample_atom_once(simulation::Simulation)
    Bₙ, Bₑ = (simulation.BₙBₑ_strength == "CQD") ? (1.1884177310293015e-5, 0.05580626719338844) : (12.36e-3, 58.12)
    θₑ₀ = (simulation.initial_m_S == -1/2) ? 0.0 : float(π)
    ϕₑ₀ = 0.0
    pool = length(simulation.initial_μₙ_sampling_weight) == 4 ? collect(-3/2:3/2) : [-3/2, 3/2]
    mᵢ = StatsBase.sample(pool, StatsBase.ProbabilityWeights(simulation.initial_μₙ_sampling_weight))
    Bₙ /= (abs(mᵢ) == 1/2) ? 3 : 1
    θₙ₀ = mᵢ > 0 ? 0.0 : float(π)
    ϕₙ₀ = 0.0
    return [θₑ₀, θₙ₀, ϕₑ₀, ϕₙ₀, Bₙ, Bₑ, mᵢ]
end

function sample_atoms(simulation::Simulation)
    return [sample_atom_once(simulation) for _ ∈ 1:simulation.atom_number]
end

"""
    is_flipped(angles::Union{Vector{Float64}, Vector{Vector{Float64}}}, simulation::Simulation)

Determine whether an atom has flipped based on its angles, magnetic field computation method, and the branching condition.
"""
function is_flipped(angles::Union{Vector{Float64}, Vector{Vector{Float64}}}, mᵢ::Float64, simulation::Simulation)
    angles = angles isa Vector{Float64} ? [angles] : angles
    j, mᵢi, pool = length(simulation.initial_μₙ_sampling_weight) == 2 ? (1/2, mᵢ / 3, [-3/2, 3/2]) : (3/2, mᵢ, collect(-3/2:3/2))
    flips = zeros(length(angles))
    for i in eachindex(angles)
        _, θₙf, _, _ = angles[i]
        CCQ_weight = [WignerD.wignerdjmn(j, mᵢi, k, θₙf)^2 for k in -j:j]
        mᵢf = StatsBase.sample(pool, StatsBase.ProbabilityWeights(CCQ_weight))
        if simulation.branching_condition == "original"
            θ_comparison = (simulation.initial_m_S == -1/2) ? mᵢf > 0 : mᵢf < 0
        else
            error("The revised branching condition has not been implemented.")
        end
        flips[i] = (simulation.magnetic_field_computation_method == "exact") ? θ_comparison : !θ_comparison
    end
    return length(flips) == 1 ? flips[1] : flips
end

"""
    get_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, simulation::Simulation)

Calculate the magnetic field components `Bx`, `By`, `Bz` at a given time `t` and return a tuple.

# Notes
- +y is right, +z is up, +x is out of page.
- Due to a different definition of the sign of current, the expressions have an extra minus sign.
"""
function get_magnetic_fields(t::Float64, current::Float64, experiment::Experiment, simulation::Simulation)
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
    Bz += simulation.B_SG * (1 / (1 + exp(-(y - simulation.y_SG) * 1e3)) + 1 / (1 + exp((y + simulation.y_SG) * 1e3)))
    return Bx, By, Bz
end

"""
    wrap(θ)

Wrap the angle θ to between 0 and π.
"""
function wrap(θ)
    return π .- abs.(mod.(θ, 2π) .- π)
end

"""
    CQD_Bloch_equation!(du, u, p, t)

Define the differential equation using CQD Bloch equations.

# Arguments
`u = [θₑ, θₙ, ϕₑ, ϕₙ]`: The variable of the differential equation.
`p = [experiment::Experiment, simulation::Simulation, current, Bₙ, Bₑ]`: The parameters passed to the solver.
"""
function CQD_Bloch_equation!(du, u, p, t)
    experiment, simulation, current, Bₙ, Bₑ = p
    θₑ, θₙ, ϕₑ, ϕₙ = u
    Bx, By, Bz = get_magnetic_fields(t, current, experiment, simulation)
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
    if θₑ_wrapped >= simulation.δθ && π - θₑ_wrapped >= simulation.δθ
        cot_θₑ = cos_θₑ / sin_θₑ
        du₃ = -γₑ * (Bz + Bₙ * cos_θₙ - cot_θₑ * (Bx * cos_ϕₑ + By * sin_ϕₑ + Bₙ_sin_θₙ * cos(Δϕₙₑ)))
        dϕₑ = du₃ - sign(du₃) * simulation.kᵢ * abs(du₁) * csc(θₑ)
    end
    if θₙ_wrapped >= simulation.δθ && π - θₙ_wrapped >= simulation.δθ
        cot_θₙ = cos_θₙ / sin_θₙ
        du₄ = -γₙ * (Bz + Bₑ * cos_θₑ - cot_θₙ * (Bx * cos_ϕₙ + By * sin_ϕₙ + Bₑ_sin_θₑ * cos(Δϕₑₙ)))
        dϕₙ = du₄ - sign(du₄) * simulation.kᵢ * abs(du₂) * csc(θₙ)
    end
    dθₑ = du₁ - sign(θₙ - θₑ) * simulation.kᵢ * abs(du₃) * sin_θₑ
    dθₙ = du₂ - sign(θₑ - θₙ) * simulation.kᵢ * abs(du₄) * sin_θₙ
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
    simulate(experiment::Experiment, simulation::Simulation)

Simulate the whole system.
"""
function simulate(experiment::Experiment, simulation::Simulation)
    raw_data = falses(length(experiment.current), simulation.atom_number)
    θₑ_plot, θₙ_plot, θₑθₙ_plot = Plots.plot(), Plots.plot(), Plots.plot()
    average_amount = 1/16
    for i ∈ eachindex(experiment.current)
        current_i = experiment.current[i]
        print("current=$i: ")
        atoms = sample_atoms(simulation)
        u₀ = atoms[1][1:4]
        ode_prob = ODEProblem(CQD_Bloch_equation!, u₀, experiment.time_span, (experiment, simulation, current_i, atoms[1][5], atoms[1][6]))
        ensemble_prob = EnsembleProblem(ode_prob, prob_func = (prob, i, repeat) -> remake(prob, u0 = atoms[i][1:4], p = (experiment, simulation, current_i, atoms[i][5], atoms[i][6])))
        @time solution = solve(ensemble_prob, simulation.solver, EnsembleDistributed(), trajectories=simulation.atom_number, reltol=1e-9, abstol=1e-9, dtmin=1e-30, force_dtmin=true, maxiters=1e14, saveat=2e-8, dt=1e-30)
        for j ∈ 1:simulation.atom_number
            mᵢ = atoms[j][7]
            sol = solution[j]
            ϕₑf, ϕₙf = sol.u[end - 1][3], sol.u[end - 1][4]
            step_number = length(sol.t)
            start_index = step_number - trunc(Int, step_number * average_amount)
            average_index_range = start_index:(step_number - 1)
            raw_data[i, j] = begin
                if simulation.average_method == "angle then branching"
                    θₑfs, θₙfs, ϕₑfs, ϕₙfs = wrap([sol.u[k][1] for k in average_index_range]), wrap([sol.u[k][2] for k in average_index_range]), [sol.u[k][3] for k in average_index_range], [sol.u[k][4] for k in average_index_range]
                    is_flipped([mean(θₑfs), mean(θₙfs), mean(ϕₑfs), mean(ϕₙfs)], mᵢ, simulation)
                elseif simulation.average_method == "branching then angle"
                    θₑfs, θₙfs, ϕₑfs, ϕₙfs = wrap([sol.u[k][1] for k in average_index_range]), wrap([sol.u[k][2] for k in average_index_range]), [sol.u[k][3] for k in average_index_range], [sol.u[k][4] for k in average_index_range]
                    angles = [collect(x) for x in zip(θₑfs, θₙfs, ϕₑfs, ϕₙfs)]
                    flips = is_flipped(angles, mᵢ, simulation)
                    flip_mean = mean(flips)
                    flip_mean >= 0.5
                else
                    θₑf, θₙf, ϕₑf, ϕₙf = wrap(sol.u[end - 1][1]), wrap(sol.u[end - 1][2]), sol.u[end - 1][3], sol.u[end - 1][4]
                    is_flipped([θₑf, θₙf, ϕₑf, ϕₙf], mᵢ, simulation)
                end
            end
        end
        sol = solution[1]
        step_number = length(sol.t)
        θₑfs, θₙfs = wrap([sol.u[k][1] for k in 1:step_number]), wrap([sol.u[k][2] for k in 1:step_number])
        Plots.plot!(θₑ_plot, sol.t * 1e6, θₑfs, label="\$$current_i\$ A", linestyle=:solid, dpi=600)
        Plots.xlabel!("Time \$t\$ [\$\\mu\$s]"); Plots.ylabel!("\$\\theta_e(t)\$"); Plots.title!("\$\\theta_e(t)\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); Plots.ylims!(0, π)
        Plots.plot!(θₙ_plot, sol.t * 1e6, θₙfs, label="\$$current_i\$ A", linestyle=:dash, dpi=600)
        Plots.xlabel!("Time \$t\$ [\$\\mu\$s]"); Plots.ylabel!("\$\\theta_n(t)\$"); Plots.title!("\$\\theta_n(t)\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); Plots.ylims!(0, π)
        Plots.plot!(θₑθₙ_plot, sol.t * 1e6, θₑfs, label="e, \$$current_i\$", linestyle=:solid, color=i)
        Plots.plot!(θₑθₙ_plot, sol.t * 1e6, θₙfs, label="n", linestyle=:dash, color=i, dpi=600)
        Plots.plot!(θₑθₙ_plot, legendcolumns=2)
        Plots.xlabel!("Time \$t\$ [\$\\mu\$s]"); Plots.ylabel!("Angles [rad]"); Plots.title!("Evolution of Angles, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$"); Plots.ylims!(0, π)
    end
    return raw_data, θₑ_plot, θₙ_plot, θₑθₙ_plot
end

"""
    struct Result

A `Result` represents the results of the simulation.

# Fields
- `raw_data::BitArray`: The raw simulation data of whether the atoms flip.
- `flip_probability::Vector{Float64}`: The flip probability calculated from the simulation.
- `R2::Float64`: The R sqaure value calculated from the simulation and experiment.
- `θₑ_plot::Plots.Plot`: The plot of θₑ dynamics.
- `θₙ_plot::Plots.Plot`: The plot of θₙ dynamics.
- `θₑθₙ_plot::Plots.Plot`: The plot that combines the two θ plots.
- `flip_plot::Plots.Plot`: The plot of the simulation and experiment flip probabilities.
"""
struct Result
    raw_data::BitArray
    flip_probability::Vector{Float64}
    R2::Float64
    θₑ_plot::Plots.Plot
    θₙ_plot::Plots.Plot
    θₑθₙ_plot::Plots.Plot
    flip_plot::Plots.Plot
    function Result(raw_data::BitArray, flip_probability::Vector{Float64}, R2::Float64, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot, flip_plot::Plots.Plot)
        length(flip_plot.series_list[1][:x]) == size(raw_data, 1) || throw(ArgumentError("The flip plot does not match the raw data in terms of the number of currents."))
        flip_plot.series_list[1][:y] == flip_probability || throw(ArgumentError("The flip plot does not match the flip probability data."))
        new(raw_data, flip_probability, R2, θₑ_plot, θₙ_plot, θₑθₙ_plot, flip_plot)
    end
end

"""
    Result(experiment::Experiment, simulation::Simulation, raw_data::BitArray, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot)

Outer constructor for `Result`.
"""
function Result(experiment::Experiment, simulation::Simulation, raw_data::BitArray, θₑ_plot::Plots.Plot, θₙ_plot::Plots.Plot, θₑθₙ_plot::Plots.Plot)
    function compute_flip_probability_and_R2(raw_data, experiment_data)
        flip_number = dropdims(sum(raw_data, dims=2), dims=2)
        flip_probability = flip_number ./ size(raw_data, 2)
        term₁ = sum((experiment_data .- Statistics.mean(experiment_data)).^2)
        term₂ = sum((experiment_data .- flip_probability).^2)
        R2 = 1 - term₂ / term₁
        return flip_probability, R2
    end
    function plot_flip_probability(experiment::Experiment, simulation::Simulation, flip_probability, R2)
        flip_plot = Plots.plot()
        Plots.plot!(flip_plot, abs.(experiment.current), flip_probability, marker=(:circle, 4), label="CQD BE Simulation")
        Plots.plot!(flip_plot, abs.(experiment.current), experiment.flip_probability, marker=(:circle, 4), label=experiment.name, dpi=600, minorgrid=true, xscale=:log10)
        Plots.xlabel!("Current [A]"); Plots.ylabel!("Flip Probability"); Plots.title!("\$R^2=" * string(trunc(R2, digits=3)) * "\$, \$k_i=" * latex_exponential(simulation.kᵢ) * "\$")
        text = "\$m_S=$(simulation.initial_m_S)\$\nInitial condition\$=\$$(round.(simulation.initial_μₙ_sampling_weight, digits=2))\n\$B_n\$ corrected Majorana\$=\$true"
        Plots.annotate!(0.07, 0.8, (text, :left, 10))
        return flip_plot
    end
    flip_probability, R2 = compute_flip_probability_and_R2(raw_data, experiment.flip_probability)
    flip_plot = plot_flip_probability(experiment, simulation, flip_probability, R2)
    return Result(raw_data, flip_probability, R2, θₑ_plot, θₙ_plot, θₑθₙ_plot, flip_plot)
end

"""
    save_result(experiment::Experiment, simulation::Simulation, result::Result, start_time, file_dir)

Save the results of the whole simulation.
"""
function save_result(experiment::Experiment, simulation::Simulation, result::Result, start_time, file_dir)
    folder_dir = joinpath(file_dir, Dates.format(start_time, "yyyy-mm-dd_HH-MM-SS.sss"))
    isdir(folder_dir) || mkdir(folder_dir)
    df1 = DataFrames.DataFrame(result.raw_data, :auto)
    CSV.write(joinpath(folder_dir, "Raw_Data.csv"), df1)
    df2 = DataFrames.DataFrame(result.flip_probability[:, :], :auto)
    CSV.write(joinpath(folder_dir, "Flip_Probability.csv"), df2)
    Plots.savefig(result.θₑ_plot, joinpath(folder_dir, "θe_Plot.svg"))
    Plots.savefig(result.θₙ_plot, joinpath(folder_dir, "θn_Plot.svg"))
    Plots.savefig(result.θₑθₙ_plot, joinpath(folder_dir, "θeθn_Plot.svg"))
    Plots.savefig(result.flip_plot, joinpath(folder_dir, "Flip_Plot.svg"))
    cp(@__FILE__, joinpath(folder_dir, "BECQDBase.jl"))
    end_time = Dates.now()
    metadata = DataStructures.OrderedDict(
        "Experiment" => experiment.name,
        "Number of Atoms" => simulation.atom_number,
        "Magnetic Field Computation Method" => simulation.magnetic_field_computation_method,
        "Initial m_S" => simulation.initial_m_S,
        "Initial μₙ Sample Probability Weight" => simulation.initial_μₙ_sampling_weight,
        "Lande g Ratio Index" => simulation.g_ratio_index,
        "θₙ" => "vary",
        "Bₙ Bₑ Strength" => simulation.BₙBₑ_strength,
        "Branching Condition" => simulation.branching_condition,
        "Average Method" => simulation.average_method,
        "kᵢ" => simulation.kᵢ,
        "Simulation Start Time" => Dates.format(start_time, "yyyy-mm-dd HH:MM:SS.sss"),
        "Simulation End Time" => Dates.format(end_time, "yyyy-mm-dd HH:MM:SS.sss"),
        "Simulation Run Time" => string(Dates.canonicalize(end_time - start_time)),
        "δθ" => simulation.δθ,
        "BE Solver Coordinates" => "spherical",
        "Remnant Fields [T]" => experiment.Bᵣ,
        "Atom Speed [m/s]" => experiment.v,
        "zₐ [m]" => experiment.zₐ,
        "Differential Equations Solver" => nameof(typeof(simulation.solver)),
        "Flight Path Length [mm]" => experiment.system_length * 1e3,
        "Flight Time Range [μs]" => experiment.time_span .* 1e6,
        "Wire Currents [A]" => experiment.current,
        "Experiment Flip Probability" => experiment.flip_probability,
        "Simulation Flip Probability" => result.flip_probability,
        "R2" => result.R2,
        "θ Cross Detection On" => simulation.θ_cross_is_detected
    )
    if simulation.θ_cross_is_detected
        metadata["θ Cross Detection Method"] = simulation.θ_cross_detection_method
        metadata["Detection Start Time [s]"] = simulation.detection_start_time
        metadata["Detection Period [s]"] = simulation.detection_period
    end
    if simulation.B_SG > 0
        metadata["B_SG [T]"] = simulation.B_SG
        metadata["y_SG [m]"] = simulation.y_SG
    end
    merge!(metadata, DataStructures.OrderedDict(
        "Machine" => Sys.MACHINE,
        "CPU" => Sys.cpu_info()[1].model,
        "Total Memory [GB]" => Sys.total_memory() / 1024^3,
        "Julia Version" => VERSION
    ))
    open(joinpath(folder_dir, "Info.json"), "w") do file
        JSON3.pretty(file, JSON3.write(metadata))
        println(file)
    end
    pkg_info = Pkg.dependencies()
    pkg_data = DataStructures.OrderedDict()
    for (uuid, info) in pkg_info
        pkg_data[info.name] = info.version
    end
    open(joinpath(folder_dir, "Pkg_Info.json"), "w") do file
        JSON3.pretty(file, JSON3.write(pkg_data))
        println(file)
    end
end

end