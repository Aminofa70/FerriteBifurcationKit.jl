using FerriteBifurcationKit
using Test

#=
To run the test, in terminal, we use :
julia --project=. -e 'using Pkg; Pkg.test()'

=#
# @testset "FerriteBifurcationKit.jl" begin
#     # Write your tests here.
# end

@testset "test_nr_disp_control.jl" begin
    include("test_nr_disp_control.jl")
end 

@testset "test_PALC_disp_control.jl" begin
    include("test_PALC_disp_control.jl")
end