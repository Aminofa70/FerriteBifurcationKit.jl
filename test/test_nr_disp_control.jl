using BifurcationKit
using Comodo
using ComodoFerrite
using Ferrite

@testset "test_newton_raphson_disp_control.jl" begin
    boxDim = [10, 10, 10]
    boxEl = [1, 1, 1]
    E, V, F, Fb, Cb = hexbox(boxDim, boxEl)
    grid = ComodoToFerrite(E, V)

    Fb_bottom = Fb[Cb .== 1]
    addface!(grid, "bottom", Fb_bottom)

    Fb_front = Fb[Cb .== 3]
    addface!(grid, "front", Fb_front)

    Fb_top = Fb[Cb .== 2]
    addface!(grid, "top", Fb_top)

    Fb_left = Fb[Cb .== 6]
    addface!(grid, "left", Fb_left)

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

    function create_dofhandler(grid)
        dh = Ferrite.DofHandler(grid)
        Ferrite.add!(dh, :u, Ferrite.Lagrange{Ferrite.RefHexahedron,1}()^3)
        Ferrite.close!(dh)
        return dh
    end

    function create_bc(dh)
        ch = Ferrite.ConstraintHandler(dh)

        dbc = Dirichlet(:u, getfacetset(dh.grid, "bottom"), (x, t) -> [0.0], [3])
        add!(ch, dbc)

        dbc = Dirichlet(:u, getfacetset(dh.grid, "front"), (x, t) -> [0.0], [2])
        add!(ch, dbc)

        dbc = Dirichlet(:u, getfacetset(dh.grid, "left"), (x, t) -> [0.0], [1])
        add!(ch, dbc)

        dbc = Dirichlet(:u, getfacetset(dh.grid, "top"), (x, t) -> [t], [3])
        add!(ch, dbc)

        Ferrite.close!(ch)
        Ferrite.update!(ch, 0.0)

        return ch
    end

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

    function constitutive_driver(C, mp::NeoHooke)
        ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
        S = 2.0 * ∂Ψ∂C
        ∂S∂C = 2.0 * ∂²Ψ∂C²
        return S, ∂S∂C
    end

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

        return
    end

    cell_values, facet_values = create_values()
    dh = create_dofhandler(grid)
    ch = create_bc(dh)


    E_mod = 1.0
    ν = 0.4
    μ = E_mod / (2 * (1 + ν))
    λ = (E_mod * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    K = allocate_matrix(dh)

    par = (mp=mp, dh=dh, cv=cell_values, ch=ch, K=K, t=0.0)


    function Fres(u, p)
        (; dh, cv, mp, ch, K, t) = p
        g = zeros(eltype(u), ndofs(dh))
        assemble_global!(K, g, dh, cv, mp, u)
        Ferrite.update!(ch, t)
        apply_zero!(g, ch)
        return g
    end

    function Jac(u, p)
        (; dh, cv, mp, ch, K, t) = p
        g = zeros(eltype(u), ndofs(dh))
        assemble_global!(K, g, dh, cv, mp, u)
        Ferrite.update!(ch, t)
        apply!(K, ch)
        return K
    end

    u0 = zeros(ndofs(dh))

    prob = BifurcationProblem(Fres, u0, par, (@optic _.t);
        J=Jac,
        record_from_solution=(x, p; k...) -> (nrm=norm(x),))

    optnewton = NewtonPar(tol=1e-8, max_iterations=25, verbose=true,
        linsolver=DefaultLS())
    u_total = 1.0                     # total prescribed displacement of "top" in z
    nsteps = 10
    ts = range(u_total / nsteps, u_total; length=nsteps)

    u = zeros(ndofs(dh))
    sols = Vector{Vector{Float64}}()

    for (k, tk) in enumerate(ts)
        Ferrite.update!(ch, tk)
        apply!(u, ch)

        probk = BifurcationKit.re_make(prob; u0=copy(u), params=(@set par.t = tk))

        sol = BifurcationKit.solve(probk, Newton(), optnewton)
    
        sol.converged || error("Newton did not converge at step $k (t = $tk)")

        u .= sol.u
        push!(sols, copy(u))
        println("step $k / $nsteps :  t = $(round(tk, digits=4))   iters = $(sol.itnewton)")
    end

    @test u[1] ≈ 0.0 atol=1e-5 
    @test u[4] ≈ -0.37683375229912125 atol=1e-5 
    @test u[23] ≈ -0.37683375229912125 atol=1e-5
    @test u[end] ≈ 1.0 atol=1e-5
end # end of test