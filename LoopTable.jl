using JSON3, DataFrames, XLSX

function process_summary(folder)
    summary_file = joinpath(folder, "Summary.json")
    if isfile(summary_file)
        json_data = JSON3.read(summary_file, Dict)
        json_data["Initial μₙ"] = json_data["Initial μₙ [relative to local B_ext]"]
        pop!(json_data, "Initial μₙ [relative to local B_ext]")
        json_data["Sigmoid Field"] = json_data["Sigmoid Field"] isa String ? json_data["Sigmoid Field"] : "($(json_data["Sigmoid Field"][1]), $(json_data["Sigmoid Field"][2]))"
        json_data["Average Method"] = json_data["Average Method"] isa String ? json_data["Average Method"] : "($(json_data["Average Method"][1]), $(json_data["Average Method"][2]))"
        return json_data
    else
        error("Summary.json not found in folder: $folder")
    end
end
folders = filter(isdir, readdir(pwd(), join=true))
rows = []
for folder in folders
    json_data = process_summary(folder)
    push!(rows, json_data)
end
df = DataFrame(rows)
df = df[:, Symbol.(["Initial μₙ", "θₙ", "Bₙ Ratio", "Branching Condition", "Average Method", "Sigmoid Field", "Max R2"])]
sorted_df = sort(df, Symbol("Max R2"), rev=true)
XLSX.writetable("Loop_Results.xlsx", sorted_df)
println("Excel file 'Loop_Results.xlsx' created successfully.")