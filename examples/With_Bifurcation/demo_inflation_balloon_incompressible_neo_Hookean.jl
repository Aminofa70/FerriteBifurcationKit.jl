using BifurcationKit
using Comodo
using Comodo.GLMakie
using Comodo.GLMakie.Colors
using Comodo.GeometryBasics
using Comodo.Statistics
using ComodoFerrite
using Ferrite
using BlockArrays
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
Hyperelastic strain energy function (neo-Hookean)
=#
struct NeoHooke
    μ::Float64
end

function Ψ(F, p, mp::NeoHooke)
    μ = mp.μ
    Ic = tr(tdot(F))
    J = det(F)
    return μ / 2 * (Ic - 3) + p * (J - 1)
end
#=
Gradient and Hessian of strain enenrgy function to get stress and tangent
=#
function constitutive_driver(F, p, mp::NeoHooke)
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, p, mp), F, :all)
    ∂²Ψ∂p², ∂Ψ∂p = Tensors.hessian(y -> Ψ(F, y, mp), p, :all)
    ∂²Ψ∂F∂p = Tensors.gradient(q -> Tensors.gradient(y -> Ψ(y, q, mp), F), p)
    return ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, ∂²Ψ∂p², ∂²Ψ∂F∂p
end
#=
Interpolation space for the shape functions
Quadrature points
=#
function create_values(interpolation_u, interpolation_p)
    qr = QuadratureRule{RefHexahedron}(3)
    facet_qr = FacetQuadratureRule{RefHexahedron}(3)
    cellvalues_u = CellValues(qr, interpolation_u)
    facetvalues_u = FacetValues(facet_qr, interpolation_u)
    cellvalues_p = CellValues(qr, interpolation_p)
    return cellvalues_u, cellvalues_p, facetvalues_u
end
#=
Deegrees of freedom
=#
function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    add!(dh, :u, ipu)
    add!(dh, :p, ipp)
    close!(dh)
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
Local assembling of residual and tangent stiffness
=#
function assemble_element!(
        Ke, fe, cell, cellvalues_u, cellvalues_p, fv,
        mp, ue, pe, ΓN, pressure
    )
    ublock, pblock = 1, 2
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)
    fill!(Ke, 0.0)
    fill!(fe, 0.0)
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, ue)
        p  = function_value(cellvalues_p, qp, pe)
        F  = one(∇u) + ∇u

        ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, _, ∂²Ψ∂F∂p = constitutive_driver(F, p, mp)

        for i in 1:nu
            ∇δui = shape_gradient(cellvalues_u, qp, i)
            fe[BlockIndex((ublock), (i))] += (∇δui ⊡ ∂Ψ∂F) * dΩ

            for j in 1:nu
                ∇δuj = shape_gradient(cellvalues_u, qp, j)
                Ke[BlockIndex((ublock, ublock), (i, j))] += ((∇δui ⊡ ∂²Ψ∂F²) ⊡ ∇δuj) * dΩ
            end

            for j in 1:np
                δp = shape_value(cellvalues_p, qp, j)
                Ke[BlockIndex((ublock, pblock), (i, j))] += (∂²Ψ∂F∂p ⊡ ∇δui) * δp * dΩ
            end
        end

        for i in 1:np
            δp = shape_value(cellvalues_p, qp, i)
            fe[BlockIndex((pblock), (i))] += δp * ∂Ψ∂p * dΩ

            for j in 1:nu
                ∇δuj = shape_gradient(cellvalues_u, qp, j)
                Ke[BlockIndex((pblock, ublock), (i, j))] += (∇δuj ⊡ ∂²Ψ∂F∂p) * δp * dΩ
            end
        end
    end

    # ── follower pressure: u-block only ───────────────────────
    for facet in 1:nfacets(cell)
        if FacetIndex(cellid(cell), facet) in ΓN
            reinit!(fv, cell, facet)
            for q_point in 1:getnquadpoints(fv)
                ∇u    = function_gradient(fv, q_point, ue)
                F     = one(∇u) + ∇u
                J     = det(F)
                FinvT = inv(F)'
                pvec  = -pressure * getnormal(fv, q_point)
                T0    = J * (FinvT ⋅ pvec)
                dΓ0   = getdetJdV(fv, q_point)

                for i in 1:nu
                    δui = shape_value(fv, q_point, i)
                    fe[BlockIndex((ublock), (i))] -= (δui ⋅ T0) * dΓ0

                    for j in 1:nu
                        δF    = shape_gradient(fv, q_point, j)
                        term1 = (FinvT ⊡ δF) * (FinvT ⋅ pvec)
                        term2 = FinvT ⋅ (δF' ⋅ (FinvT ⋅ pvec))
                        δT0   = J * (term1 - term2)
                        Ke[BlockIndex((ublock, ublock), (i, j))] -= (δui ⋅ δT0) * dΓ0
                    end
                end
            end
        end
    end
end
#=
Global assembling of residual and tangent stiffness
=#
function assemble_global!(K, f, cellvalues_u, cellvalues_p, fv, dh, mp, w, ΓN, pressure)
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    fe = BlockedArray(zeros(nu + np), [nu, np])
    ke = BlockedArray(zeros(nu + np, nu + np), [nu, np], [nu, np])

    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        dofs = celldofs(cell)
        assemble_element!(
            ke, fe, cell,
            cellvalues_u, cellvalues_p, fv,
            mp, w[dofs[1:nu]], w[dofs[(nu + 1):end]],
            ΓN, pressure
        )
        assemble!(assembler, dofs, ke, fe)
    end
end
#=
Finite Element values and material parameters
=#
ipu = Lagrange{RefHexahedron, 2}()^3           
ipp = Lagrange{RefHexahedron, 1}()            

cvu, cvp, fv = create_values(ipu, ipp)      
dh = create_dofhandler(grid, ipu, ipp)
ch = create_bc(dh)
K  = allocate_matrix(dh)

ΓN = getfacetset(grid, "inner")

μ_mod = 1.0
mp = NeoHooke(μ_mod)

par = (dh = dh, cvu = cvu, cvp = cvp, fv = fv,
       mp = mp, ch = ch, K = K, ΓN = ΓN, pressure = 0.0)

function Fres(w, p)
    (; dh, cvu, cvp, fv, mp, ch, K, ΓN, pressure) = p
    g = zeros(eltype(w), ndofs(dh))
    assemble_global!(K, g, cvu, cvp, fv, dh, mp, w, ΓN, pressure)
    apply_zero!(g, ch)
    return g
end

function Jac(w, p)
    (; dh, cvu, cvp, fv, mp, ch, K, ΓN, pressure) = p
    g = zeros(eltype(w), ndofs(dh))
    assemble_global!(K, g, cvu, cvp, fv, dh, mp, w, ΓN, pressure)
    apply!(K, ch)
    return copy(K)
end

w0 = zeros(ndofs(dh))
for cell in CellIterator(dh)
    w0[celldofs(cell)[dof_range(dh, :p)]] .= -μ_mod
end


prob = BifurcationProblem(Fres, w0, par, (@optic _.pressure); J = Jac)
prob = BifurcationProblem(Fres, w0, par, (@optic _.pressure); J = Jac)

optnewton = NewtonPar(tol = 1e-4, max_iterations = 100, verbose = true, linsolver = DefaultLS())
optcont = ContinuationPar(
    p_min = 0.0, p_max = 0.4,
    ds = 0.002, dsmin = 1.0e-9, dsmax = 0.02,
    max_steps = 100,
    newton_options = optnewton,
    detect_bifurcation = 0,
    save_sol_every_step = 1,
)

r_ref  = [norm(V[i]) for i in eachindex(V)]
λ_stop = 2.0                                   # target stretch

function stretch_of(w)
    un = vec(evaluate_at_grid_nodes(dh, w, :u))
    return maximum(norm(V[i] .+ un[i]) / r_ref[i] for i in eachindex(V))
end

br = continuation(prob, PALC(), optcont;
    normC = norminf, verbosity = 2,
    finalise_solution = (z, tau, step, contResult; k...) -> stretch_of(z.u) < λ_stop)

# br = continuation(prob, PALC(), optcont; normC = norminf, verbosity = 2)

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
lines!(ax3, stretch, press, color = :black, linewidth = 2.6)
hp3 = scatter!(ax3, [Point2f(stretch[stepStart], press[stepStart])]; markersize = 15, color = :red)


k  = rIn / rOut
λi = range(1.0, maximum(stretch); length = 300)
λo = @. (1 + k^3 * (λi^3 - 1))^(1 / 3)
p_analytic = @. 2 * μ_mod * ((1 / λo + 1 / (4 * λo^4)) - (1 / λi + 1 / (4 * λi^4)))

scatter!(ax3, λi, p_analytic, color = :red, markersize = 2)

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