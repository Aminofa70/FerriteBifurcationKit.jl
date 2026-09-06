using BifurcationKit
using Comodo
using ComodoFerrite
using Ferrite

@testset "test_newton_raphson_disp_control.jl" begin
    boxDim = [10, 10, 10]
    boxEl = [1, 1, 1]

    Eh, V, Fh, Fb, Cb = hexbox(boxDim, boxEl)
    grid = ComodoToFerrite(Eh, V)

    addface!(grid, "bottom", Fb[Cb .== 1])
    addface!(grid, "top", Fb[Cb .== 2])
    addface!(grid, "front", Fb[Cb .== 3])
    addface!(grid, "left", Fb[Cb .== 6])


    function create_values()
        order, dim = 1, 3
        ip = Lagrange{RefHexahedron,order}()^dim
        qr = QuadratureRule{RefHexahedron}(2)
        qr_face = FacetQuadratureRule{RefHexahedron}(1)
        return CellValues(qr, ip), FacetValues(qr_face, ip)
    end

    function create_dofhandler(grid)
        dh = Ferrite.DofHandler(grid)
        Ferrite.add!(dh, :u, Ferrite.Lagrange{Ferrite.RefHexahedron,1}()^3)
        Ferrite.close!(dh)
        return dh
    end

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

    struct NeoHooke
        μ::Float64
        λ::Float64
    end

    function Ψ(C, mp::NeoHooke)
        Ic = tr(C)
        J = sqrt(det(C))
        return mp.μ / 2 * (Ic - 3) - mp.μ * log(J) + mp.λ / 2 * (log(J))^2
    end

    function constitutive_driver(C, mp::NeoHooke)
        ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
        return 2.0 * ∂Ψ∂C, 2.0 * ∂²Ψ∂C²
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
        return
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

    Emod = 1.0
    ν = 0.4
    mp = NeoHooke(Emod / (2 * (1 + ν)), (Emod * ν) / ((1 + ν) * (1 - 2ν)))

    K = allocate_matrix(dh)
    par = (mp=mp, dh=dh, cv=cell_values, ch=ch, K=K, t=0.0)

    function Fres(u, p)
        (; dh, cv, mp, ch, K, t) = p
        g = zeros(eltype(u), ndofs(dh))
        assemble_global!(K, g, dh, cv, mp, u)

        Ferrite.update!(ch, t)
        ū = zeros(ndofs(dh))
        apply!(ū, ch)                       # ū[d] = prescribed value at this t

        for d in ch.prescribed_dofs
            g[d] = u[d] - ū[d]
        end
        return g
    end

    function Jac(u, p)
        (; dh, cv, mp, ch, K) = p
        g = zeros(eltype(u), ndofs(dh))
        assemble_global!(K, g, dh, cv, mp, u)

        for d in ch.prescribed_dofs
            K[d, :] .= 0.0
            K[d, d] = 1.0
        end
        return copy(K)                      # copy: BifKit keeps references to it
    end


    u0 = zeros(ndofs(dh))

    prob = BifurcationProblem(Fres, u0, par, (@optic _.t);
        J=Jac,
        record_from_solution=(x, p; k...) -> (nrm=norm(x), umax=maximum(abs, x)))

    @assert norm(Fres(u0, par)) < 1e-10 "u = 0 at t = 0 must be an exact solution"

    optnewton = NewtonPar(tol=1e-8, max_iterations=20, verbose=false,
        linsolver=DefaultLS())

    optcont = ContinuationPar(
        p_min=0., p_max=1.0,      # range of the prescribed top displacement
        ds=0.05,
        dsmin=1e-6, dsmax=0.25,
        max_steps=300,
        newton_options=optnewton,
        detect_bifurcation=0,
        save_sol_every_step=1,
    )

    br = continuation(prob, PALC(), optcont; normC=norminf, verbosity=2)

    u = br.sol[end].x

    @test u[1] ≈ 0.0 atol=1e-5
    @test u[4] ≈ -0.3768337523966147 atol=1e-5
    @test u[23] ≈ -0.37683375239661454 atol=1e-5
    @test u[end] ≈ 1.0 atol=1e-5

end # end of test