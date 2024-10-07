cwd = pwd()
folders = filter(isdir, readdir(cwd, join=true))
for folder ∈ folders
    subfolders = filter(isdir, readdir(folder, join=true))
    for subfolder ∈ subfolders
        new_location = joinpath(cwd, basename(subfolder))
        mv(subfolder, new_location)
    end
    rm(folder, recursive=true)
end
println("Loop went back.")