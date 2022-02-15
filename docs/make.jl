using FastGMM
using Documenter

DocMeta.setdocmeta!(FastGMM, :DocTestSetup, :(using FastGMM); recursive=true)

makedocs(;
    modules=[FastGMM],
    authors="Andreas Hildebrandt <andreas.hildebrandt@uni-mainz.de> and contributors",
    repo="https://github.com/hildebrandtlab/FastGMM.jl/blob/{commit}{path}#{line}",
    sitename="FastGMM.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hildebrandtlab.github.io/FastGMM.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hildebrandtlab/FastGMM.jl",
    devbranch="main",
)
