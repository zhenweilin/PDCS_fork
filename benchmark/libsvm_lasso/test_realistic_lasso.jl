using Test
using SparseArrays
using JuMP
using LinearAlgebra

include(joinpath(@__DIR__, "realistic_lasso.jl"))
using .RealisticLasso

function test_spec(path)
    return DatasetSpec(
        "tiny",
        "tiny",
        path,
        "",
        :xz,
        nothing,
        3,
        4,
        filesize(path),
        :pm_one,
        "test",
        "",
        "",
    )
end

@testset "real-data Lasso parser and model" begin
    mktempdir() do dir
        plain = joinpath(dir, "tiny.libsvm")
        compressed = joinpath(dir, "tiny.libsvm.xz")
        open(plain, "w") do io
            println(io, "1 1:2.0 3:1.0")
            println(io, "0 2:3.0 4:-1.0")
            println(io, "-1 1:1.0 4:2.0")
        end
        run(pipeline(`xz -c $plain`, stdout = compressed))
        spec = test_spec(compressed)

        scan = scan_libsvm(spec; progress_every = 0)
        @test scan.rows == 3
        @test scan.features == 4
        @test scan.nnz == 6
        @test scan.lambda_reference == 3.0
        @test scan.lambda_zero_threshold == 6.0

        limited_scan = scan_libsvm(spec; max_rows = 1, progress_every = 0)
        @test limited_scan.rows == 1
        @test limited_scan.nnz == 2
        @test limited_scan.lambda_reference == 2.0

        data = load_libsvm(spec; progress_every = 0)
        expected_A = sparse(
            [1, 3, 2, 1, 2, 3],
            [1, 1, 2, 3, 4, 4],
            Float32[2, 1, 3, 1, -1, 2],
            3,
            4,
        )
        @test data.A == expected_A
        @test data.b == Float32[1, -1, -1]
        @test penalties(data, [1.0, 0.1]) ≈ [3.0, 0.3]

        built = build_lasso_socp(data; penalty_ratio = 0.1)
        @test built.lambda ≈ 0.3
        @test JuMP.num_variables(built.model) == 2 * 4 + 1
        @test JuMP.objective_sense(built.model) == JuMP.MOI.MIN_SENSE
        @test set_penalty_ratio!(built, data, 1.0) == 3.0
        @test built.lambda == 3.0
        @test JuMP.objective_function(built.model).terms[built.u[1]] == 3.0
        @test get(JuMP.objective_function(built.model).terms, built.x[1], 0.0) == 0.0

        bulk = build_lasso_conic_data(
            data;
            penalty_ratio = 0.1,
            workers = 1,
        )
        @test bulk.formulation == :compact_epigraph_direct_soc
        @test bulk.num_variables == 2 * size(data.A, 2) + 1
        @test bulk.num_rows == size(data.A, 1) + 2 * size(data.A, 2) + 2
        @test length(bulk.nzval) == nnz(data.A) + 4 * size(data.A, 2) + 2
        @test eltype(bulk.colptr) == Int64
        @test eltype(bulk.rowval) == Int64
        @test map(block -> block.set_type, bulk.cone_blocks) == [
            JuMP.MOI.Nonnegatives,
            JuMP.MOI.SecondOrderCone,
        ]
        @test map(block -> block.dimension, bulk.cone_blocks) == [
            2 * size(data.A, 2),
            size(data.A, 1) + 2,
        ]

        G = SparseMatrixCSC(
            bulk.num_rows,
            bulk.num_variables,
            bulk.colptr,
            bulk.rowval,
            bulk.nzval,
        )
        x = [0.25, -0.5, 0.0, 0.75]
        t = abs.(x)
        residual = Float64.(data.A) * x - Float64.(data.b)
        r = dot(residual, residual) / 2.0
        point = vcat(x, t, r)
        image = G * point + bulk.affine_constants
        m, n = size(data.A)
        soc_first = 2 * n + 1
        @test image[1:n] ≈ t - x
        @test image[(n + 1):(2 * n)] ≈ t + x
        @test image[(soc_first + 2):end] ≈ residual
        @test image[soc_first] ≈ norm(image[(soc_first + 1):end])
        @test dot(bulk.objective_coefficients, point) ≈
              2.0 * r + bulk.penalty * sum(abs, x)

        threaded_bulk = build_lasso_conic_data(
            data;
            penalty_ratio = 0.1,
            workers = 2,
        )
        @test threaded_bulk.colptr == bulk.colptr
        @test threaded_bulk.rowval == bulk.rowval
        @test threaded_bulk.nzval == bulk.nzval
        @test_throws ArgumentError build_lasso_conic_data(data; workers = 0)

        @test set_penalty!(bulk, 2.5) == 2.5
        @test bulk.penalty == 2.5
        @test bulk.objective_coefficients[1:n] == zeros(n)
        @test bulk.objective_coefficients[(n + 1):(2 * n)] == fill(2.5, n)
        @test bulk.objective_coefficients[end] == 2.0

        bulk_objective_model = Model()
        @variable(bulk_objective_model, z[1:bulk.num_variables])
        @objective(bulk_objective_model, Min, sum(z))
        @test set_penalty!(bulk_objective_model, bulk, 4.0) == 4.0
        bulk_objective = JuMP.objective_function(bulk_objective_model)
        for column in bulk.x_range
            @test get(bulk_objective.terms, z[column], 0.0) == 0.0
        end
        for column in bulk.epigraph_range
            @test bulk_objective.terms[z[column]] == 4.0
        end
        @test bulk_objective.terms[z[bulk.r_index]] == 2.0

        compact_spec = DatasetSpec(
            spec.id,
            spec.display_name,
            spec.path,
            spec.url,
            spec.container,
            spec.archive_member,
            spec.expected_rows,
            6,
            spec.expected_bytes,
            spec.label_mode,
            spec.source,
            spec.official_page,
            spec.notes,
        )
        compact = load_libsvm(
            compact_spec;
            compact_zero_columns = true,
            progress_every = 0,
        )
        @test size(compact.A) == (3, 4)
        @test compact.A == expected_A
        @test compact.original_features == 6

        archive_root = joinpath(dir, "archive")
        archive_member = joinpath("criteo.kaggle2014.svm", "train.txt.svm")
        archive_source = joinpath(archive_root, archive_member)
        mkpath(dirname(archive_source))
        cp(plain, archive_source)
        archive_path = joinpath(dir, "tiny.tar.xz")
        run(`tar -cJf $archive_path -C $archive_root $archive_member`)
        archive_spec = DatasetSpec(
            "tiny-archive",
            "tiny archive",
            archive_path,
            "",
            :tar_xz,
            archive_member,
            3,
            4,
            filesize(archive_path),
            :pm_one,
            "test",
            "",
            "",
        )
        archive_scan = scan_libsvm(archive_spec; progress_every = 0)
        @test archive_scan.rows == 3
        @test archive_scan.nnz == 6
        @test archive_scan.lambda_reference == 3.0
    end
end
