using JSON3, Plots, LaTeXStrings

function categorize_subfolders(base_dir)
    subfolders = filter(isdir, readdir(base_dir, join=true))
    groups = Dict{String, Vector{String}}()
    for subfolder ∈ subfolders
        json_file = joinpath(subfolder, "Info.json")
        json_data = JSON3.read(json_file, Dict)
        subfolder_key = Dict{Any, Any}("θₙ" => json_data["θₙ"],
        "Branching Condition" => json_data["Branching Condition"],
        "Average Method" => json_data["Average Method"],
        "Initial μₙ [relative to local B_ext]" => json_data["Initial μₙ [relative to local B_ext]"],
        "Sigmoid Field" => json_data["Sigmoid Field"],
        "Bₙ Ratio" => json_data["Bₙ Ratio"])
        subfolder_key_string = JSON3.write(subfolder_key)
        if haskey(groups, subfolder_key_string)
            push!(groups[subfolder_key_string], subfolder)
        else
            groups[subfolder_key_string] = [subfolder]
        end
    end
    for (group_key, key_values) ∈ groups
        group_key_data = JSON3.read(group_key, Dict)
        new_folder_name = joinpath(base_dir, "$(hash(group_key))")
        mkpath(new_folder_name)
        for subfolder ∈ key_values
            dest_folder = joinpath(new_folder_name, basename(subfolder))
            mv(subfolder, dest_folder)
        end
        summary_file = joinpath(new_folder_name, "Summary.json")
        open(summary_file, "w") do f
            JSON3.pretty(f, JSON3.write(group_key_data))
            println(f)
        end
    end
end

function plot_kᵢ(base_dir)
    subfolders = filter(isdir, readdir(base_dir, join=true))
    simulation_flip_probabilities = Vector{Float64}[]
    simulation_flip_probabilities_stds = Vector{Float64}[]
    kᵢ = Float64[]
    max_R2_index = 1
    for (i, subfolder) ∈ enumerate(subfolders)
        json_file = joinpath(subfolder, "Info.json")
        json_data = JSON3.read(json_file, Dict)
        if i == 1
            global experiment_name = json_data["Experiment"]
            global experiment_currents = json_data["Wire Currents [A]"]
            global experiment_flip_probabilities = json_data["Experiment Flip Probabilities"]
            global experiment_flip_probabilities_stds = replace(json_data["Experiment Flip Probabilities Standard Deviations"], nothing => NaN)
            global qm_flip_probabilities = json_data["QM Flip Probabilities"]
            global max_R2 = json_data["R2"]
        else
            if json_data["Experiment"] != experiment_name
                error("The experiments of the simulations are different.")
            end
            if json_data["R2"] > max_R2
                max_R2 = json_data["R2"]
                max_R2_index = i
            end
        end
        push!(simulation_flip_probabilities, json_data["Simulation Flip Probabilities"])
        push!(simulation_flip_probabilities_stds, json_data["Simulation Flip Probabilities Standard Deviations"])
        push!(kᵢ, json_data["kᵢ"])
    end
    function latex_exponential(n)
        if n == 0
            return "0"
        else
            exponent = floor(Int, log10(abs(n)))
            mantissa = n / 10.0^exponent
            if mantissa == 1.0
                return "10^{$exponent}"
            else
                return "$mantissa\\times10{$exponent}"
            end
        end
    end
    Plots.default(fontfamily="Computer Modern", tickfont=10, linewidth=1.5, framestyle=:box, legendfontsize=9)
    plot_range = split(experiment_name, " ")[1] == "FS" ? (2:lastindex(experiment_currents)) : (1:lastindex(experiment_currents))
    total_plot = plot()
    scatter!(total_plot, abs.(experiment_currents[plot_range]), experiment_flip_probabilities[plot_range], yerr = experiment_flip_probabilities_stds[plot_range], marker=(:xcross, 6), markerstrokewidth=3, linewidth=2, label=experiment_name, legend=:best, markerstrokecolor=:auto, dpi=600, minorgrid=true, xscale=:log10)
    scatter!(total_plot, abs.(experiment_currents[plot_range]), qm_flip_probabilities[plot_range], marker=(:plus, 6), markerstrokewidth=3, label="QM")
    for k in eachindex(simulation_flip_probabilities)
        Plots.plot!(total_plot, abs.(experiment_currents[plot_range]), simulation_flip_probabilities[k][plot_range], yerr = simulation_flip_probabilities_stds[k][plot_range], marker=(:circle, 4), markerstrokecolor=:auto, label="\$k_i=" * latex_exponential(kᵢ[k]) * "\$")
    end
    function suffix(n)
        if n == 1
            return "st"
        elseif n == 2
            return "nd"
        elseif n == 3
            return "rd"
        else
            return "th"
        end
    end
    xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("\$R^2_\\mathrm{max}=" * string(trunc(max_R2, digits=3)) * "\$ (\$k_i=" * latex_exponential(kᵢ[max_R2_index]) * "\$)"); ylims!(0, 1)
    savefig(total_plot, joinpath(base_dir, "Total_Plot.svg"))
    best_plot = plot()
    scatter!(best_plot, abs.(experiment_currents[plot_range]), experiment_flip_probabilities[plot_range], yerr = experiment_flip_probabilities_stds[plot_range], marker=(:xcross, 6), markerstrokewidth=3, linewidth=2, label=experiment_name, legend=:best, markerstrokecolor=:auto, dpi=600, minorgrid=true, xscale=:log10)
    scatter!(best_plot, abs.(experiment_currents[plot_range]), qm_flip_probabilities[plot_range], marker=(:plus, 6), markerstrokewidth=3, label="QM")
    plot!(best_plot, abs.(experiment_currents[plot_range]), simulation_flip_probabilities[max_R2_index][plot_range], yerr = simulation_flip_probabilities_stds[max_R2_index][plot_range], marker=(:circle, 4), markerstrokecolor=:auto, label="\$k_i=" * latex_exponential(kᵢ[max_R2_index]) * "\$")
    xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("\$R^2=" * string(trunc(max_R2, digits=3)) * "\$"); ylims!(0, 1)
    savefig(best_plot, joinpath(base_dir, "Best_Plot.svg"))
    return max_R2
end

categorize_subfolders(pwd())
subfolders = filter(isdir, readdir(pwd(), join=true))
max_R2s = []
for subfolder ∈ subfolders
    push!(max_R2s, plot_kᵢ(subfolder))
    json_file = joinpath(subfolder, "Summary.json")
    json_data = JSON3.read(json_file, Dict)
    json_data["Max R2"] = max_R2s[end]
    open(json_file, "w") do f
        JSON3.pretty(f, JSON3.write(json_data))
        println(f)
    end
end
rank = sortperm(max_R2s, rev=true)
order = similar(rank)
order[rank] = collect(1:length(rank))
for (i, subfolder) ∈ enumerate(subfolders)
    mv(subfolder, joinpath(dirname(subfolder), "$(order[i])"))
end
println("Loop process finished.")