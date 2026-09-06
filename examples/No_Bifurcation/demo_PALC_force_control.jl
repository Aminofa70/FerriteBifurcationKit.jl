using BifurcationKit
using Comodo
using Comodo.GLMakie
using Comodo.GLMakie.Colors
using Comodo.GeometryBasics
using Comodo.Statistics
using ComodoFerrite
using Ferrite
using LinearAlgebra
## GLMakie setting
GLMakie.closeall()

#=
Geometry and mesh using Comodo.jl
=#
boxDim = [10, 10, 10]
boxEl = [10, 10, 10]
E, V, F, Fb, Cb = hexbox(boxDim, boxEl)
#=
Connecting Comodo mesh to Ferrite Grid style using ComodoFerrite.jl
Adding the faces from Comodo mesh to Ferrite
=#
grid = ComodoToFerrite(E, V)
addface!(grid, "bottom", Fb[Cb .== 1])
addface!(grid, "front", Fb[Cb .== 3])
addface!(grid, "top", Fb[Cb .== 2])
addface!(grid, "left", Fb[Cb .== 6])
#=
Interpolation space for the shape functions
Quadrature points
=#
function create_values()
    order = 1
    dim = 3
    ip = Lagrange{RefHexahedron,order}()^dim
    qr = QuadratureRule{RefHexahedron}(2)
    qr_face = FacetQuadratureRule{RefHexahedron}(1)
    cell_values = CellValues(qr, ip)
    facet_values = FacetValues(qr_face, ip)
    return cell_values, facet_values
end

#=
Deegrees of freedom
=#
function create_dofhandler(grid)
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, Ferrite.Lagrange{Ferrite.RefHexahedron,1}()^3)
    Ferrite.close!(dh)
    return dh
end
#=
Dirichlet Boundary Condition
=#
function create_bc(dh)
    ch = Ferrite.ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "bottom"), (x, t) -> [0.0], [3]))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "front"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "left"), (x, t) -> [0.0], [1]))
    Ferrite.close!(ch)
    Ferrite.update!(ch, 0.0)
    return ch
end
#=
Hyperelastic strain energy function (neo-Hookean)
=#
struct NeoHooke
    μ::Float64
    λ::Float64
end
function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * (log(J))^2
end
#=
Gradient and Hessian of strain enenrgy function to get stress and tangent
=#
function constitutive_driver(C, mp::NeoHooke)
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end
#=
Local assembling of residual and tangent stiffness
=#
function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, tn)
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(I, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I)
        for i in 1:ndofs
            ∇δui = shape_gradient(cv, qp, i)
            ge[i] += (∇δui ⊡ P) * dΩ
            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
            end
        end
    end
    # loop for the traction load
    for facet in 1:nfacets(cell)
        if (cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                T0 = tn
                dΓ0 = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ T0) * dΓ0
                end
            end
        end
    end
end;
#=
Global assembling of residual and tangent stiffness
=#
function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, tn)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    assembler = start_assemble(K, g)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] 
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, tn)
        assemble!(assembler, global_dofs, ke, ge)
    end
end
#=
Finite Element values and material parameters
=#
ΓN = getfacetset(grid, "top")
cell_values, facet_values = create_values()
dh = create_dofhandler(grid)
ch = create_bc(dh)
K = allocate_matrix(dh)

Emod = 10.0
ν    = 0.3
mp   = NeoHooke(Emod / (2 * (1 + ν)), (Emod * ν) / ((1 + ν) * (1 - 2ν)))

par = (mp = mp, dh = dh, cv = cell_values, fv = facet_values, ch = ch, K = K, ΓN = ΓN, t = 0.0)


#=
Residual function for BifurcationKit.jl
=#
function Fres(u, p)
    (; dh, cv, fv, mp, ch, K, ΓN, t) = p
    g  = zeros(eltype(u), ndofs(dh))
    tn = Ferrite.Vec{3}((0.0, 0.0, t))
    assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, tn)
    apply_zero!(g, ch)
    return g
end
#=
Jacobian function for BifurcationKit.jl
=#
function Jac(u, p)
    (; dh, cv, fv, mp, ch, K, ΓN, t) = p
    g  = zeros(eltype(u), ndofs(dh))
    tn = Ferrite.Vec{3}((0.0, 0.0, t))
    assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, tn)
    apply!(K, ch)
    return copy(K)
end
#=
BifurcationProblem  (parameter = t, the prescribed top displacement)
=#
u0 = zeros(ndofs(dh))
prob = BifurcationProblem(Fres, u0, par, (@optic _.t);
    J = Jac,
    record_from_solution = (x, p; k...) -> (nrm = norm(x), umax = maximum(abs, x)))

@assert norm(Fres(u0, par)) < 1e-10 "u = 0 at t = 0 must be an exact solution"
optnewton = NewtonPar(tol=1e-8, max_iterations=25, verbose=true,linsolver=DefaultLS())
optcont = ContinuationPar(
    p_min = 0., p_max = 1.0,        # traction range (same units as traction_max)
    ds    = 0.05,                   
    dsmin = 1e-6, dsmax = 0.25,
    max_steps = 300,
    newton_options = optnewton,
    detect_bifurcation = 0,        
    save_sol_every_step = 1,    
)
br = continuation(prob, PALC(), optcont; normC = norminf, verbosity = 2)

nsteps = length(br.sol) 
nd = ndofs(dh)
UT         = Vector{Vector{Point{3,Float64}}}(undef, nsteps)
UT_mag     = Vector{Vector{Float64}}(undef, nsteps)
ut_mag_max = zeros(Float64, nsteps)

for (k, s) in enumerate(br.sol)
    u_nodes     = vec(evaluate_at_grid_nodes(dh, s.x, :u))
    disp_points = [Point{3,Float64}([un[1], un[2], un[3]]) for un in u_nodes]
    UT[k ]         = disp_points
    UT_mag[k]     = norm.(disp_points)
    ut_mag_max[k] = maximum(UT_mag[k])
end
#=
plot the displacement using GLMakie and Comodo
=#
numInc = length(UT)
scale = 1.0
VT = [V .+ scale .* UT[i] for i in 1:numInc]
min_p = minp([minp(Vi) for Vi in VT])
max_p = maxp([maxp(Vi) for Vi in VT])
incRange = 0:(numInc - 1)

fig = Figure(size = (800,600))
stepStart = 1

ax1 = AxisGeom(fig[1, 1], title = "Step: $stepStart", limits = (min_p[1], max_p[1], min_p[2], max_p[2], min_p[3], max_p[3]) )
hp1 = meshplot!(ax1, Fb, VT[stepStart + 1]; strokewidth = 2, color = UT_mag[stepStart + 1], transparency = false, colormap = Reverse(:Spectral), colorrange = (0.0, maximum(ut_mag_max)))
Colorbar(fig[1, 2], hp1.plots[1], label = "Displacement magnitude [mm]")
hSlider = Slider(fig[2, :], range = incRange, startvalue = stepStart, linewidth = 30)
on(hSlider.value) do stepIndex
    i = stepIndex + 1
    hp1[1] = GeometryBasics.Mesh(VT[i], F)
    hp1.color = UT_mag[i]
    ax1.title = "Step: $stepIndex"
end
slidercontrol(hSlider, ax1)
screen = display(GLMakie.Screen(), fig)