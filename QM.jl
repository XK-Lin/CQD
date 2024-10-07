using LinearAlgebra, Statistics, Plots, DifferentialEquations, ODEInterfaceDiffEq, WignerD, QuantumOptics
include("CQDBase.jl")
import .CQDBase: Experiment, get_external_magnetic_fields

const ħ = 6.62607015e-34/(2π)
const μ₀ = 4π * 1e-7
const γₑ = -1.76085963023e11
const γₙ = 1.25e7
const a_hfs = 230.8598601e6
const I = 3/2
const S = 1/2

initial_state = -1/2
magnetic_field_computation_method = "exact"
sigmoid_field = "off"
# experiment = Experiment("FS Low \$z_a\$")
experiment = Experiment(
    "FS Low \$z_a\$ with By",
    -1 * [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
    [0.000001275588301, 0.011491017802886, 0.059235010197955, 0.147571912832191, 0.224749989856136, 0.278763884568857, 0.258084512687681, 0.159140733524888, 0.002466693134262],
    [0.000002551176603, 0.012706621114225, 0.019087441872173, 0.041843884468939, 0.065821040395060, 0.045884573918560, 0.036562847662003, 0.037966146555234, 0.004933386268525],
    [1.618181088923953e-17, 8.222328341118226e-7, 0.06413051347614414, 0.12223458274766767, 0.21266440131608902, 0.24995836392356835, 0.24999944698021592, 0.2499998869331271, 0.249999973719386],
    105e-6, 800.0, [0, 40e-6, 42e-6], 16e-3, 16e-3 / 800 .* (-0.5, 0.5)
)

βₑ = SpinBasis(Rational(S))
σx, σy, σz = Matrix.((sigmax(βₑ).data, sigmay(βₑ).data, sigmaz(βₑ).data))
βₙ = SpinBasis(Rational(I))
τx, τy, τz = Matrix.((sigmax(βₙ).data, sigmay(βₙ).data, sigmaz(βₙ).data))
nS, nI = trunc.(Int64, (2 * S + 1, 2 * I + 1))
Iₑ = Matrix{ComplexF64}(LinearAlgebra.I, nS, nS)
Iₙ = Matrix{ComplexF64}(LinearAlgebra.I, nI, nI)

Sₑx, Sₑy, Sₑz = kron.((σx, σy, σz), fill(Iₙ, 3))
Sₙx, Sₙy, Sₙz = kron.(fill(Iₑ, 3), (τx, τy, τz))

σ_int = (Sₑx * Sₙx + Sₑy * Sₙy + Sₑz * Sₙz) / 4
A_hfs = a_hfs * ħ * 2π
H_int = A_hfs * σ_int

function get_initial_ρ₀(state::Union{String, Float64})
    if state ∈ ("up", -1/2)
        ρₑ₀ = [0 0; 0 1]
    else
        ρₑ₀ = [1 0; 0 0]
    end
    return kron(ρₑ₀, Iₙ / 4)
end

function get_hamiltonian(B)
    Hₑ = -γₑ * ħ / 2 * kron(sum(B .* (σx, σy, σz)), Iₙ)
    Hₙ = -γₙ * ħ / 2 * kron(Iₑ, sum(B .* (τx, τy, τz)))
    return Hₑ + Hₙ + H_int
end

function vonNeumann!(du, u, p, t)
    # p = (current, experiment, magnetic_field_computation_method, sigmoid_field)
    # u[:, 1:nS * nI]: real(kron(ρₙ, ρₑ))
    # u[:, nS * nI + 1:2 * nS * nI]: imag(kron(ρₙ, ρₑ))
    B = get_external_magnetic_fields(t, p...)
    H = get_hamiltonian(B)
    ρ = u[:, 1:nS * nI] + 1im * u[:, nS * nI + 1:2 * nS * nI]
    dρ = (H * ρ - ρ * H) / (1im * ħ)
    du .= hcat(real.(dρ), imag.(dρ))
end

function get_eigenstates(t, current, experiment, magnetic_field_computation_method, sigmoid_field)
    eigenstates = eigvecs(get_hamiltonian(get_external_magnetic_fields(t, current, experiment, magnetic_field_computation_method, sigmoid_field)))
    return eigenstates ./ norm.(eachcol(eigenstates))
end

flip_probabilities = []
for i ∈ eachindex(experiment.currents)
    v = get_eigenstates(experiment.time_span[1], experiment.currents[i], experiment, magnetic_field_computation_method, sigmoid_field)
    ρ₀ = get_initial_ρ₀(initial_state)
    ρ₀ = v * ρ₀ * v'
    u₀ = hcat(real.(ρ₀), imag.(ρ₀))
    p = (experiment.currents[i], experiment, magnetic_field_computation_method, sigmoid_field)
    prob = ODEProblem(vonNeumann!, u₀, experiment.time_span, p)
    @time solution = solve(prob, radau5(), reltol=1e-6, abstol=1e-6, dtmin=1e-30, force_dtmin=true, maxiters=1e14, saveat=2e-8, dt=1e-30)
    ρend = solution.u[end][:, 1:nS * nI] + 1im * solution.u[end][:, nS * nI + 1:2 * nS * nI]
    v = get_eigenstates(solution.t[end], experiment.currents[i], experiment, magnetic_field_computation_method, sigmoid_field)
    ρend = v' * ρend * v
    diagonal = abs.(diag(ρend))
    if initial_state ∈ ("up", -1/2)
        p_i = sum(diagonal[1:trunc(Int, nS * nI / 2)])
    else
        p_i = sum(diagonal[trunc(Int, nS * nI / 2 + 1):end])
    end
    push!(flip_probabilities, p_i)
end

qmplot = Plots.plot()
plot!(qmplot, abs.(experiment.currents[2:end]), flip_probabilities[2:end], marker=(:circle, 4), label="QM for " * experiment.name, legend=:best, dpi=600, minorgrid=true, xscale=:log10)
xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("QM Simulation")