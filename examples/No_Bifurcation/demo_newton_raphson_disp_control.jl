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
    cell_values = CellValues(qr, ip)
    return cell_values
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
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "top"), (x, t) -> [t], [3]))
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
function assemble_element!(ke, ge, cell, cv, mp, ue)
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
end
#=
Global assembling of residual and tangent stiffness
=#
function assemble_global!(K, g, dh, cv, mp, u)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)
    assembler = start_assemble(K, g)
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs]
        assemble_element!(ke, ge, cell, cv, mp, ue)
        assemble!(assembler, global_dofs, ke, ge)
    end
end
#=
Finite Element values and material parameters
=#
cell_values = create_values()
dh = create_dofhandler(grid)
ch = create_bc(dh)
K = allocate_matrix(dh)

E_mod = 1.0
ν = 0.4
μ = E_mod / (2 * (1 + ν))
λ = (E_mod * ν) / ((1 + ν) * (1 - 2ν))
mp = NeoHooke(μ, λ)
#=
Parametrs for BifurcationKit.jl
=#
par = (mp=mp, dh=dh, cv=cell_values, ch=ch, K=K, t=0.0)
#=
Residual function for BifurcationKit.jl
=#
function Fres(u, p)
    (; dh, cv, mp, ch, K, t) = p
    g = zeros(eltype(u), ndofs(dh))
    assemble_global!(K, g, dh, cv, mp, u)
    Ferrite.update!(ch, t)
    apply_zero!(g, ch)                  # residual ≡ 0 on prescribed dofs
    return g
end
#=
Jacobian function for BifurcationKit.jl
=#
function Jac(u, p)
    (; dh, cv, mp, ch, K, t) = p
    g = zeros(eltype(u), ndofs(dh))
    assemble_global!(K, g, dh, cv, mp, u)
    Ferrite.update!(ch, t)
    apply!(K, ch)                       # zero rows/cols, 1 on the diagonal
    return K
end
#=
BifurcationProblem  (parameter = t, the prescribed top displacement)
=#
u0 = zeros(ndofs(dh))
prob = BifurcationProblem(Fres, u0, par, (@optic _.t); J=Jac, record_from_solution=(x, p; k...) -> (nrm=norm(x),))
optnewton = NewtonPar(tol=1e-8, max_iterations=25, verbose=true,linsolver=DefaultLS())
#=
Solving the problem and save displacement
=#
u_total = 1.0
nsteps  = 10
ts      = range(u_total / nsteps, u_total; length = nsteps)

nd = ndofs(dh)
UT         = Vector{Vector{Point{3,Float64}}}(undef, nsteps + 1)
UT_mag     = Vector{Vector{Float64}}(undef, nsteps + 1)
ut_mag_max = zeros(Float64, nsteps + 1)

u    = zeros(nd)
sols = Vector{Vector{Float64}}()

# ── step 0: t = 0, undeformed ────────────────────────────────
u_nodes     = vec(evaluate_at_grid_nodes(dh, u, :u))
disp_points = [Point{3,Float64}([un[1], un[2], un[3]]) for un in u_nodes]

UT[1]         = disp_points
UT_mag[1]     = norm.(disp_points)
ut_mag_max[1] = maximum(UT_mag[1])
push!(sols, copy(u))

# ── steps 1 … nsteps ─────────────────────────────────────────
for (k, tk) in enumerate(ts)
    Ferrite.update!(ch, tk)
    apply!(u, ch)

    probk = BifurcationKit.re_make(prob; u0 = copy(u), params = (@set par.t = tk))
    sol   = BifurcationKit.solve(probk, Newton(), optnewton)
    sol.converged || error("Newton did not converge at step $k (t = $tk)")

    u .= sol.u
    push!(sols, copy(u))
    println("step $k / $nsteps :  t = $(round(tk, digits = 4))   iters = $(sol.itnewton)")

    u_nodes     = vec(evaluate_at_grid_nodes(dh, u, :u))
    disp_points = [Point{3,Float64}([un[1], un[2], un[3]]) for un in u_nodes]

    UT[k + 1]         = disp_points
    UT_mag[k + 1]     = norm.(disp_points)
    ut_mag_max[k + 1] = maximum(UT_mag[k + 1])
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