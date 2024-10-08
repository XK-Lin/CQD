include("CQDDataAnalysisBase.jl")
using .CQDDataAnalysisBase

cwd = pwd()
for dir ∈ filter(isdir, readdir(cwd, join=true))
    categorize_folders(dir, ("θₙ", "Branching Condition", "Average Method", "Initial μₙ [relative to local B_ext]", "Sigmoid Field", "Bₙ Bₑ Ratio"))
    sort_folders(dir)
    get_subfolder_summary_tables(dir)
end
clean_folders(cwd)