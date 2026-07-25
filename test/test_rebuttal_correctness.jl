using Test
include("../benchmark/rebuttal/validate_projection.jl")
using .ValidateProjection

@testset "SOC validation" begin
    input=[2.0,1.0,0.0, -2.0,1.0,0.0, 0.0,2.0,0.0]
    output=copy(input)
    for r in (1:3,4:6,7:9)
        project_soc!(@view output[r])
    end
    result=validate_soc(input,output,3,3)
    @test result.status=="PASS"
    broken=copy(output); broken[1]+=1
    @test validate_soc(input,broken,3,3).status=="FAIL"
end
