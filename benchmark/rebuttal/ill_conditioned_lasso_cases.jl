module IllConditionedLassoCases

using LinearAlgebra
using Random
using SHA
using SparseArrays
using Statistics

export LassoInstance, generate_instance, verify_instance, lasso_metrics,
       instance_hashes, kappa_target, rho_for_kappa

struct LassoInstance
    A::SparseMatrixCSC{Float64,Int}
    b::Vector{Float64}
    xstar::Vector{Float64}
    rstar::Vector{Float64}
    support::Vector{Int}
    lambda::Float64
    K::Float64
    rho::Float64
    panel::Symbol
    seed::Int
    sparsity::Int
    retries::Int
end

rho_for_kappa(K::Real) = (Float64(K)-1)/(Float64(K)+1)
kappa_target(rho::Real) = (1+rho)/(1-rho)

function _unit_rademacher!(I,V,rows,rng)
    d=length(rows); scale=inv(sqrt(Float64(d)))
    for row in rows
        push!(I,row); push!(V,rand(rng,Bool) ? scale : -scale)
    end
end

function _draw_xstar(s,rng)
    q=Vector{Float64}(undef,s); amplitude=Vector{Float64}(undef,s)
    for j in 1:2:s
        sign=rand(rng,Bool) ? 1.0 : -1.0
        q[j]=sign; q[j+1]=-sign
        amplitude[j]=1+rand(rng); amplitude[j+1]=1+rand(rng)
    end
    x=amplitude.*q; x./=norm(x)
    x,q
end

"""Create a sparse paired-column Lasso instance with an exactly controlled
active Gram condition number.  The inactive columns are pivot-corrected so
their correlation with the known optimal residual is zero."""
function generate_instance(m::Integer,n::Integer,s::Integer,d::Integer,K::Real,
                           seed::Integer; panel::Symbol=:fixed_lambda,
                           lambda::Real=1e-2, residual_norm::Real=1.0)
    m>0 && n>s && iseven(s) && s>0 || error("require m>0, n>s>0, and even s")
    d>0 && m>=s*d || error("exact active construction needs m >= s*d")
    K>=1 || error("K must be at least one")
    panel in (:fixed_lambda,:fixed_residual) || error("unknown panel")
    rng_active=MersenneTwister(seed+0x5100)
    rng_inactive=MersenneTwister(seed+0x6200)
    rho=rho_for_kappa(K)
    xS,q=_draw_xstar(s,MersenneTwister(seed+0x7300))

    # Build active columns in unpermuted coordinates.  Each pair occupies two
    # disjoint d-row supports, hence different pairs are orthogonal exactly.
    I=Int[]; J=Int[]; V=Float64[]
    sizehint!(I,Int(3s*d÷2 + (n-s)*(d+1)))
    sizehint!(J,length(I)); sizehint!(V,length(I))
    for pair in 1:s÷2
        urows=collect((2pair-2)*d+1:(2pair-1)*d)
        vrows=collect((2pair-1)*d+1:2pair*d)
        uvalues=[rand(rng_active,Bool) ? inv(sqrt(Float64(d))) : -inv(sqrt(Float64(d))) for _ in urows]
        vvalues=[rand(rng_active,Bool) ? inv(sqrt(Float64(d))) : -inv(sqrt(Float64(d))) for _ in vrows]
        c1=2pair-1; c2=2pair
        append!(I,urows); append!(J,fill(c1,d)); append!(V,uvalues)
        append!(I,urows); append!(J,fill(c2,d)); append!(V,rho.*uvalues)
        append!(I,vrows); append!(J,fill(c2,d)); append!(V,sqrt(max(0,1-rho^2)).*vvalues)
    end
    Aactive=sparse(I,J,V,m,s)
    coeff=Vector{Float64}(undef,s)
    for j in 1:2:s
        # Inverse of each 2x2 Gram block applied to opposite signs.
        coeff[j]=q[j]/(1-rho); coeff[j+1]=q[j+1]/(1-rho)
    end
    h=Aactive*coeff
    λ=panel===:fixed_lambda ? Float64(lambda) : 2Float64(residual_norm)/norm(h)
    rstar=-(λ/2).*h

    # Inactive sparse columns use a common seeded stream.  Their K-dependent
    # pivot correction makes their residual correlation exactly zero.
    pivot_candidates=sortperm(abs.(rstar);rev=true)[1:max(1,cld(count(!iszero,rstar),2))]
    retries=0
    for col in s+1:n
        support=randperm(rng_inactive,m)[1:d]
        values=randn(rng_inactive,d)
        pivot=first(filter(p->!(p in support) && rstar[p]!=0,pivot_candidates))
        correction=-dot(rstar[support],values)/rstar[pivot]
        norm2=sqrt(sum(abs2,values)+correction^2)
        append!(I,support); append!(J,fill(col,d)); append!(V,values./norm2)
        push!(I,pivot); push!(J,col); push!(V,correction/norm2)
    end
    A=sparse(I,J,V,m,n)
    # A single common row permutation avoids visible contiguous active blocks.
    rowperm=randperm(MersenneTwister(seed+0x8400),m)
    A=A[rowperm,:]; rstar=rstar[rowperm]
    xstar=vcat(xS,zeros(n-s))
    b=A*xstar-rstar
    LassoInstance(A,b,xstar,rstar,collect(1:s),λ,Float64(K),rho,panel,
      Int(seed),Int(d),retries)
end

function lasso_metrics(inst::LassoInstance,x::AbstractVector{<:Real})
    r=inst.A*x-inst.b; at_r=inst.A' * r
    τ=1e-10*max(1,norm(x,Inf))
    stationarity=0.0
    for j in eachindex(x)
        e=abs(x[j])>τ ? abs(2at_r[j]+inst.lambda*sign(x[j])) :
          max(2abs(at_r[j])-inst.lambda,0.0)
        stationarity=max(stationarity,e)
    end
    normalized=stationarity/(1+2norm(at_r,Inf)+inst.lambda)
    f=dot(r,r)+inst.lambda*sum(abs,x)
    rs=inst.rstar
    fs=dot(rs,rs)+inst.lambda*sum(abs,inst.xstar)
    found=findall(j->abs(x[j])>τ,eachindex(x)); actual=Set(inst.support)
    precision=isempty(found) ? 1.0 : count(j->j in actual,found)/length(found)
    recall=count(j->abs(x[j])>τ,inst.support)/length(inst.support)
    (;stationarity,normalized_stationarity=normalized,
      x_error=norm(x-inst.xstar)/(1+norm(inst.xstar)),
      objective=f,objective_error=abs(f-fs)/(1+abs(fs)),precision,recall,
      residual_norm=norm(r))
end

function verify_instance(inst::LassoInstance)
    S=inst.support; G=Matrix(inst.A[:,S]'*inst.A[:,S])
    eig=eigvals(Symmetric(G)); k=maximum(eig)/minimum(eig)
    target=kappa_target(inst.rho)
    active=norm(2*(inst.A[:,S]'*inst.rstar)+inst.lambda*sign.(inst.xstar[S]),Inf)
    inactive=isempty(S)==false ? maximum(abs.(2*(inst.A[:,setdiff(1:size(inst.A,2),S)]'*inst.rstar));init=0.0) : 0.0
    metrics=lasso_metrics(inst,inst.xstar)
    column_norm_error=maximum(abs.(sqrt.(sum(abs2,inst.A;dims=1)).-1))
    (;kappa_measured=k,kappa_theory=target,
      kappa_relative_error=abs(k-target)/target,kkt_active=active,
      kkt_inactive=inactive,normalized_stationarity=metrics.normalized_stationarity,
      column_norm_error)
end

function instance_hashes(inst::LassoInstance)
    hash(x)=bytes2hex(sha256(reinterpret(UInt8,vec(x))))
    # CSC internal arrays preserve the sparse numerical representation.
    (;matrix=bytes2hex(sha256(vcat(reinterpret(UInt8,inst.A.colptr),
       reinterpret(UInt8,inst.A.rowval),reinterpret(UInt8,inst.A.nzval)))),
      b=hash(inst.b),xstar=hash(inst.xstar))
end

end
