using BifurcationKit
using Comodo
using Comodo.GLMakie
using Comodo.GLMakie.Colors
using Comodo.GeometryBasics
using Comodo.Statistics
using ComodoFerrite
using Ferrite
## GLMakie setting
GLMakie.closeall()
#=
Geometry and mesh using Comodo.jl
=#
rOut = 5.0
rIn = 4.5                   
pointSpacing = 1.0
E, V = hexspherehollow(rOut, rIn, pointSpacing)
#=
Connecting Comodo mesh to Ferrite Grid style using ComodoFerrite.jl
Adding the faces from Comodo mesh to Ferrite
=#
grid = ComodoToFerrite(E, V)
F = element2faces(E)
Fb = boundaryfaces(element2faces(E))
Fc = simplexcenter(Fb, V)
rc = norm.(Fc)
rmid = 0.5 * (rIn + rOut)

addface!(grid, "inner", Fb[rc .< rmid])
addface!(grid, "outer", Fb[rc .> rmid])
#=
Apply symmetric boundary condition to avoid rigid body motion
=#
tol = 1.0e-6 * rOut
addnodeset!(grid, "symx", x -> abs(x[1]) < tol)
addnodeset!(grid, "symy", x -> abs(x[2]) < tol)
addnodeset!(grid, "symz", x -> abs(x[3]) < tol)
#=
Interpolation space for the shape functions
Quadrature points
=#
function create_values()
    order = 1
    dim = 3
    ip = Lagrange{RefHexahedron, order}()^dim
    qr = QuadratureRule{RefHexahedron}(2)
    qr_face = FacetQuadratureRule{RefHexahedron}(2)
    cv = CellValues(qr, ip)
    fv = FacetValues(qr_face, ip)
    return cv, fv
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
    add!(ch, Dirichlet(:u, getnodeset(dh.grid, "symx"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getnodeset(dh.grid, "symy"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getnodeset(dh.grid, "symz"), (x, t) -> [0.0], [3]))
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
function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, pressure)
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)
    ndofs = getnbasefunctions(cv)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
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
    # Follower traction on Neumann boundary
    for facet in 1:nfacets(cell)
        if FacetIndex(cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                ∇u = function_gradient(fv, q_point, ue)
                F = one(∇u) + ∇u
                J = det(F)
                FinvT = inv(F)'
                pressure_val = -pressure * getnormal(fv, q_point)
                T0 = J * (FinvT ⋅ pressure_val)
                dΓ0 = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ T0) * dΓ0
                    for j in 1:ndofs
                        ∇δuj = shape_gradient(fv, q_point, j)
                        δF = ∇δuj
                        term1 = (FinvT ⊡ δF) * (FinvT ⋅ pressure_val)
                        term2 = FinvT ⋅ (δF' ⋅ (FinvT ⋅ pressure_val))
                        δT0 = J * (term1 - term2)
                        ke[i, j] -= (δui ⋅ δT0) * dΓ0
                    end
                end
            end
        end
    end
end
#=
Global assembling of residual and tangent stiffness
=#
function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, pressure)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    assembler = start_assemble(K, g)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, pressure)
        assemble!(assembler, global_dofs, ke, ge)
    end
end
#=
Finite Element values and material parameters
=#
ΓN = getfacetset(grid, "inner")
cv, fv = create_values()
dh = create_dofhandler(grid)
ch = create_bc(dh)
K = allocate_matrix(dh)
μ_mod = 1.0
λ_mod = 50.0                 
mp = NeoHooke(μ_mod, λ_mod)

#=
Parametrs for BifurcationKit.jl
=#
par = (dh = dh, cv = cv, fv = fv, mp = mp, ch = ch, K = K, ΓN = ΓN, pressure = 0.0)
#=
Residual function for BifurcationKit.jl
=#
function Fres(u, p)
    (; dh, cv, fv, mp, ch, K, ΓN, pressure) = p
    g = zeros(eltype(u), ndofs(dh))
    assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, pressure)
    apply_zero!(g, ch)
    return g
end
#=
Jacobian function for BifurcationKit.jl
=#
function Jac(u, p)
    (; dh, cv, fv, mp, ch, K, ΓN, pressure) = p
    g = zeros(eltype(u), ndofs(dh))
    assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, pressure)
    apply!(K, ch)
    return copy(K)
end
#=
BifurcationProblem  (parameter = t, the prescribed top displacement)
=#
u0 = zeros(ndofs(dh))
prob = BifurcationProblem(Fres, u0, par, (@optic _.pressure);
    J = Jac,
    record_from_solution = (x, p; k...) -> (nrm = norm(x), umax = maximum(abs, x)))

@assert norm(Fres(u0, par)) < 1e-10 "u = 0 at t = 0 must be an exact solution"
optnewton = NewtonPar(tol=1e-8, max_iterations=25, verbose=true,linsolver=DefaultLS())
optcont = ContinuationPar(
    p_min = 0.0, p_max = 0.40,       
    ds = 0.002, dsmin = 1.0e-9, dsmax = 0.02,
    max_steps = 100,
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
Define radial stretch
=#
r_ref = [norm(V[i]) for i in eachindex(V)] # reference radius of every node
press   = [s.p for s in br.sol]            # pressure at each step
stretch = Float64[]
for s in br.sol
    un    = vec(evaluate_at_grid_nodes(dh, s.x, :u))            # nodal displacements
    r_def = [norm(V[i] .+ un[i]) for i in eachindex(V)]         # deformed radius
    push!(stretch, maximum(r_def ./ r_ref))
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

ax1 = AxisGeom(fig[1, 1], title = "Step: $stepStart",limits = (min_p[1], max_p[1], min_p[2], max_p[2], min_p[3], max_p[3]))
hp1 = meshplot!(ax1, Fb, VT[stepStart];strokewidth = 2, color = UT_mag[stepStart], transparency = false,colormap = Reverse(:Spectral), colorrange = (0.0, maximum(ut_mag_max)))
Colorbar(fig[1, 2], hp1.plots[1], label = "Displacement magnitude [mm]")
ax3 = Axis(fig[1, 3], title = "Step: $stepStart", aspect = AxisAspect(1), xlabel = "circumferential stretch  λ_θ", ylabel = "pressure  p")
lines!(ax3, stretch, press, color = :black, linewidth = 1.6)
hp3 = scatter!(ax3, [Point2f(stretch[stepStart], press[stepStart])]; markersize = 15, color = :red)

hSlider = Slider(fig[2, :], range = incRange, startvalue = stepStart, linewidth = 30)

on(hSlider.value) do stepIndex
    i = stepIndex + 1
    hp1[1]    = GeometryBasics.Mesh(VT[i], Fb)
    hp1.color = UT_mag[i]
    hp3[1]    = [Point2f(stretch[i], press[i])]
    ax1.title = "Step: $stepIndex"
    ax3.title = "Step: $stepIndex"
end

slidercontrol(hSlider, ax1)
screen = display(GLMakie.Screen(), fig)

