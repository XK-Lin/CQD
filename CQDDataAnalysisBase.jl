#=
CQDDataAnalysisBase.jl

This module defines functions that analyze results from CQDBase simulations.
Author: Xukun Lin
Update: 10/07/2024

Required packages: "JSON3", "Plots", "LaTeXStrings", "DataFrames", "XLSX".
=#

module CQDDataAnalysisBase

using JSON3, Plots, LaTeXStrings, DataFrames, XLSX

export unpack_folders, categorize_folders, sort_folders, get_subfolder_summary_tables, clean_folders

"""
    unpack_folders(directory::String)

For each folder in `directory`, move all subfolders out of it and delete the folder. Files in the folder that are not directories are deleted.
"""
function unpack_folders(directory::String)
    folder_paths = filter(isdir, readdir(directory, join=true))
    for folder_path ∈ folder_paths
        subfolder_paths = filter(isdir, readdir(folder_path, join=true))
        for subfolder_path ∈ subfolder_paths
            destination = joinpath(directory, basename(subfolder_path))
            mv(subfolder_path, destination)
        end
        rm(folder_path, recursive=true)
    end
    println("All folders in '$directory' are unpacked.")
end

"""
    categorize_folders(directory::String, fields::Union{String, Tuple{Vararg{String}}})

Divide all folders in `directory` into groups according to the `fields`. Then for each group, create a new folder in `directory` and move the group into it.
"""
function categorize_folders(directory::String, fields::Union{String, Tuple{Vararg{String}}})
    folder_paths = filter(isdir, readdir(directory, join=true))
    folder_groups = Dict{String, Vector{String}}()
    for folder_path ∈ folder_paths
        json_file_path = joinpath(folder_path, "Info.json")
        json_data = JSON3.read(json_file_path, Dict)
        data_for_categorization = Dict{Any, Any}(key => json_data[key] for key ∈ fields)
        data_for_categorization_string = JSON3.write(data_for_categorization)
        if haskey(folder_groups, data_for_categorization_string)
            push!(folder_groups[data_for_categorization_string], folder_path)
        else
            folder_groups[data_for_categorization_string] = [folder_path]
        end
    end
    for (group, group_folder_paths) ∈ folder_groups
        group_data_for_categorization = JSON3.read(group, Dict)
        new_group_folder_path = joinpath(directory, "$(hash(group))")
        mkpath(new_group_folder_path)
        for folder_path ∈ group_folder_paths
            new_folder_path = joinpath(new_group_folder_path, basename(folder_path))
            mv(folder_path, new_folder_path)
        end
        summary_file = joinpath(new_group_folder_path, "Summary.json")
        open(summary_file, "w") do f
            JSON3.pretty(f, JSON3.write(group_data_for_categorization))
            println(f)
        end
    end
    println("All folders in '$directory' are categorized based on the fields $fields.")
end

"""
    get_kᵢ_plots(directory::String)

Plot the data from the folders within the `directory`.
"""
function get_kᵢ_plots(directory::String)
    folder_paths = filter(isdir, readdir(directory, join=true))
    simulation_flip_probabilities_list = Vector{<:Real}[]
    simulation_flip_probabilities_stds_list = Vector{<:Real}[]
    kᵢ_list = Real[]
    max_R2_index = 1
    experiment_name = nothing
    experiment_currents = nothing
    experiment_flip_probabilities = nothing
    experiment_flip_probabilities_stds = nothing
    qm_flip_probabilities = nothing
    max_R2 = nothing
    for (i, folder_path) ∈ enumerate(folder_paths)
        json_file_path = joinpath(folder_path, "Info.json")
        json_data = JSON3.read(json_file_path, Dict)
        if i == 1
            experiment_name = json_data["Experiment"]
            experiment_currents = json_data["Wire Currents [A]"]
            experiment_flip_probabilities = json_data["Experiment Flip Probabilities"]
            experiment_flip_probabilities_stds = replace(json_data["Experiment Flip Probabilities Standard Deviations"], nothing => NaN)
            qm_flip_probabilities = json_data["QM Flip Probabilities"]
            max_R2 = json_data["R2"]
        else
            if json_data["Experiment"] != experiment_name
                println(json_data["Experiment"])
                println(experiment_name)
                error("The experiments of the simulations are different.")
            end
            if json_data["R2"] > max_R2
                max_R2 = json_data["R2"]
                max_R2_index = i
            end
        end
        push!(simulation_flip_probabilities_list, Float64.(json_data["Simulation Flip Probabilities"]))
        push!(simulation_flip_probabilities_stds_list, Float64.(json_data["Simulation Flip Probabilities Standard Deviations"]))
        push!(kᵢ_list, json_data["kᵢ"])
    end
    function latex_exponential(x::Real)
        if x == 0
            return "0"
        else
            exponent = floor(Int, log10(abs(x)))
            mantissa = x / 10.0^exponent
            return mantissa == 1 ? "10^{$exponent}" : "$mantissa\\times10{$exponent}"
        end
    end
    Plots.default(fontfamily="Computer Modern", tickfont=10, linewidth=1.5, framestyle=:box, legendfontsize=9)
    plot_range = split(experiment_name, " ")[1] == "FS" ? (2:lastindex(experiment_currents)) : (1:lastindex(experiment_currents))
    total_plot = plot()
    scatter!(total_plot, abs.(experiment_currents[plot_range]), experiment_flip_probabilities[plot_range], yerr = experiment_flip_probabilities_stds[plot_range], marker=(:xcross, 6), markerstrokewidth=3, linewidth=2, label=experiment_name, legend=:best, markerstrokecolor=:auto, dpi=600, minorgrid=true, xscale=:log10)
    scatter!(total_plot, abs.(experiment_currents[plot_range]), qm_flip_probabilities[plot_range], marker=(:plus, 6), markerstrokewidth=3, label="QM")
    for k ∈ eachindex(simulation_flip_probabilities_list)
        plot!(total_plot, abs.(experiment_currents[plot_range]), simulation_flip_probabilities_list[k][plot_range], yerr = simulation_flip_probabilities_stds_list[k][plot_range], marker=(:circle, 4), markerstrokecolor=:auto, label="\$k_i=" * latex_exponential(kᵢ_list[k]) * "\$")
    end
    function suffix(n::Int)
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
    xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("\$R^2_\\mathrm{max}=" * string(trunc(max_R2, digits=3)) * "\$ (\$k_i=" * latex_exponential(kᵢ_list[max_R2_index]) * "\$)"); ylims!(0, 1)
    savefig(total_plot, joinpath(directory, "Total_Plot.svg"))
    best_plot = plot()
    scatter!(best_plot, abs.(experiment_currents[plot_range]), experiment_flip_probabilities[plot_range], yerr = experiment_flip_probabilities_stds[plot_range], marker=(:xcross, 6), markerstrokewidth=3, linewidth=2, label=experiment_name, legend=:best, markerstrokecolor=:auto, dpi=600, minorgrid=true, xscale=:log10)
    scatter!(best_plot, abs.(experiment_currents[plot_range]), qm_flip_probabilities[plot_range], marker=(:plus, 6), markerstrokewidth=3, label="QM")
    plot!(best_plot, abs.(experiment_currents[plot_range]), simulation_flip_probabilities_list[max_R2_index][plot_range], yerr = simulation_flip_probabilities_stds_list[max_R2_index][plot_range], marker=(:circle, 4), markerstrokecolor=:auto, label="\$k_i=" * latex_exponential(kᵢ_list[max_R2_index]) * "\$")
    xlabel!("Current [A]"); ylabel!("Flip Probability"); title!("\$R^2=" * string(trunc(max_R2, digits=3)) * "\$"); ylims!(0, 1)
    savefig(best_plot, joinpath(directory, "Best_Plot.svg"))
    return max_R2
end

"""
    sort_folders(directory::String)

Sort folders in `directory` based on the max R2 value.
"""
function sort_folders(directory::String)
    folder_paths = filter(isdir, readdir(directory, join=true))
    max_R2s = []
    for folder_path ∈ folder_paths
        push!(max_R2s, get_kᵢ_plots(folder_path))
        json_file_path = joinpath(folder_path, "Summary.json")
        json_data = JSON3.read(json_file_path, Dict)
        json_data["Max R2"] = max_R2s[end]
        open(json_file_path, "w") do f
            JSON3.pretty(f, JSON3.write(json_data))
            println(f)
        end
    end
    rank = sortperm(max_R2s, rev=true)
    order = similar(rank)
    order[rank] = collect(1:length(rank))
    for (i, folder_path) ∈ enumerate(folder_paths)
        mv(folder_path, joinpath(directory, "$(order[i])"))
    end
    println("All folders in '$directory' are sorted based on max R2.")
end

"""
    get_subfolder_summary_tables(directory::String)

For each folder in `directory`, create an Excel file that contains Max-R2-ranked configurations.
"""
function get_subfolder_summary_tables(directory::String)
    function process_summary(summary_file_folder_path::String)
        summary_file_path = joinpath(summary_file_folder_path, "Summary.json")
        if isfile(summary_file_path)
            json_data = JSON3.read(summary_file_path, Dict)
            fields = keys(json_data)
            if "Initial μₙ [relative to local B_ext]" ∈ fields
                json_data["Initial μₙ"] = json_data["Initial μₙ [relative to local B_ext]"]
                pop!(json_data, "Initial μₙ [relative to local B_ext]")
            end
            if "Initial μₑ [relative to local B_ext]" ∈ fields
                json_data["Initial μₑ"] = json_data["Initial μₑ [relative to local B_ext]"]
                pop!(json_data, "Initial μₑ [relative to local B_ext]")
            end
            if "Sigmoid Field" ∈ fields
                json_data["Sigmoid Field"] = json_data["Sigmoid Field"] isa String ? json_data["Sigmoid Field"] : "($(json_data["Sigmoid Field"][1]), $(json_data["Sigmoid Field"][2]))"
            end
            if "Average Method" ∈ fields
                json_data["Average Method"] = json_data["Average Method"] isa String ? json_data["Average Method"] : "($(json_data["Average Method"][1]), $(json_data["Average Method"][2]))"
            end
            if "Bₙ Bₑ Ratio" ∈ fields
                json_data["Bₙ Bₑ Ratio"] = "($(json_data["Bₙ Bₑ Ratio"][1]), $(json_data["Bₙ Bₑ Ratio"][2]))"
            end
            return json_data
        else
            error("'Summary.json' not found in directory $summary_file_folder_path.")
        end
    end
    folder_paths = filter(isdir, readdir(directory, join=true))
    rows = []
    for folder_path in folder_paths
        json_data = process_summary(folder_path)
        push!(rows, json_data)
    end
    df = DataFrame(rows)
    function rank_columns(dataframe::DataFrame)
        column_heads = names(dataframe)
        heads_collection = ("Simulation Type", "Magnetic Field Computation Method", "Initial μₑ", "Initial μₙ", "θₙ", "Branching Condition", "Bₙ Bₑ Strength", "Bₙ Bₑ Ratio", "Average Method", "Sigmoid Field", "θ Cross Detection", "Max R2")
        collection_rank = Dict(name => i for (i, name) ∈ enumerate(heads_collection))
        if !all(in(heads_collection), column_heads)
            error("The input DataFrame contains head not in the predefined collection.")
        end
        heads_rank = [collection_rank[name] for name ∈ column_heads]
        return column_heads[sortperm(heads_rank)]
    end
    df = df[:, Symbol.(rank_columns(df))]
    sorted_df = sort(df, Symbol("Max R2"), rev=true)
    XLSX.writetable(joinpath(directory, "Loop_Results.xlsx"), sorted_df)
    println("Excel file 'Loop_Results.xlsx' created successfully in '$directory'.")
end

"""
    clean_folders(directory::String)

Clean the contents of folders in `directory` for faster upload / download.

Note that if you are using other functions from CQDDataAnalysisBase.jl in the folder `dir`, then the input for this function should be `dirname(dir)`.
"""
function clean_folders(directory::String)
    folder_paths = filter(isdir, readdir(directory, join=true))
    original_data_folder_path = mkdir(joinpath(directory, "Original Data"))
    cp.(folder_paths, joinpath.(original_data_folder_path, basename.(folder_paths)))
    allowed_subfolders = Set(["1", "2", "3", "4"])
    for folder_path ∈ folder_paths
        subfolder_paths = filter(isdir, readdir(folder_path, join=true))
        for subfolder_path ∈ subfolder_paths
            subfolder_name = basename(subfolder_path)
            if subfolder_name ∉ allowed_subfolders
                rm(subfolder_path, recursive=true)
            end
        end
        subfolder_paths = filter(isdir, readdir(folder_path, join=true))
        for subfolder_path ∈ subfolder_paths
            subsubfolder_paths = filter(isdir, readdir(subfolder_path, join=true))
            rm.(subsubfolder_paths, recursive=true)
        end
    end
    println("All folders in '$directory' are cleaned.")
end

end