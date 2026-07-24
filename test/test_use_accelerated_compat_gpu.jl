using Test
using Random
using SparseArrays
using PDCS: PDCS_CPU, PDCS_GPU

function compatibility_demo(use_accelerated)
    rng = MersenneTwister(22)
    n = 20
    m_zero = 10
    m_nonnegative = 10
    m_soc = 30
    m = m_zero + m_nonnegative + m_soc

    A = sprand(rng, m, n, 0.5)
    c = ones(n)
    feasible_x = rand(rng, n)
    b = A * feasible_x
    cone_part = copy(b)
    cone_part[1:m_zero] .= 0.0
    cone_part[m_zero + 1:m_zero + m_nonnegative] .=
        max.(cone_part[m_zero + 1:m_zero + m_nonnegative], 0.0)
    PDCS_CPU.soc_proj!(@view cone_part[m_zero + m_nonnegative + 1:end])
    b .-= cone_part

    return PDCS_GPU.rpdhg_gpu_solve(
        n=n,
        m=m,
        nb=n,
        c=c,
        G=A,
        h=b,
        mGzero=m_zero,
        mGnonnegative=m_nonnegative,
        socG=Integer[m_soc],
        rsocG=Integer[],
        expG=0,
        dual_expG=0,
        bl=zeros(n),
        bu=fill(Inf, n),
        soc_x=Integer[],
        rsoc_x=Integer[],
        use_preconditioner=true,
        use_aggressive=true,
        use_accelerated=use_accelerated,
        verbose=0,
        max_outer_iter=2,
        max_inner_iter=50,
        check_terminate_freq=10,
        duality_gap_restart_freq=10,
    )
end

@testset "deprecated use_accelerated GPU compatibility" begin
    baseline = compatibility_demo(false)
    legacy = compatibility_demo(true)

    @test baseline.info.exit_status == legacy.info.exit_status
    @test baseline.info.iter == legacy.info.iter
    @test isapprox(baseline.info.pObj, legacy.info.pObj; rtol=1e-8, atol=1e-10)
    @test isapprox(baseline.info.dObj, legacy.info.dObj; rtol=1e-8, atol=1e-10)
    @test isapprox(
        Array(baseline.x.recovered_primal.primal_sol.x),
        Array(legacy.x.recovered_primal.primal_sol.x);
        rtol=1e-8,
        atol=1e-10,
    )

    wrapper = PDCS_GPU.PDCS_GPU_Solver(
        n=1,
        m=1,
        nb=1,
        c=[1.0],
        G=zeros(1, 1),
        h=[0.0],
        mGzero=1,
        mGnonnegative=0,
        socG=Integer[],
        rsocG=Integer[],
        expG=0,
        dual_expG=0,
        bl=[-Inf],
        bu=[Inf],
        soc_x=Integer[],
        rsoc_x=Integer[],
        use_accelerated=true,
    )
    @test !hasproperty(wrapper, :use_accelerated)
end
