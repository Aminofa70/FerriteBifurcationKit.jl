using FerriteBifurcationKit
using Documenter

DocMeta.setdocmeta!(FerriteBifurcationKit, :DocTestSetup, :(using FerriteBifurcationKit); recursive=true)

makedocs(;
    modules=[FerriteBifurcationKit],
    authors="Aminofa70 <amin.alibakhshi@upm.es> and contributors",
    sitename="FerriteBifurcationKit.jl",
    format=Documenter.HTML(;
        canonical="https://Aminofa70.github.io/FerriteBifurcationKit.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Aminofa70/FerriteBifurcationKit.jl",
    devbranch="main",
)
