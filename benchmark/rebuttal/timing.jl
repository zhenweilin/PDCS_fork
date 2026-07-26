module RebuttalTiming

using CUDA
using Statistics

export cuda_event_time, repeated_cuda_time, bandwidth_lower_bound

function cuda_event_time(f)
    start = CUDA.CuEvent()
    stop = CUDA.CuEvent()
    CUDA.record(start)
    f()
    CUDA.record(stop)
    CUDA.synchronize(stop)
    1000CUDA.elapsed(start, stop)
end

function repeated_cuda_time(restore!, launch!; warmups=5, minimum_ms=100.0)
    for _ in 1:warmups
        restore!(); launch!()
    end
    CUDA.synchronize()
    restore!()
    probe = cuda_event_time(launch!)
    repetitions = max(1, ceil(Int, minimum_ms / max(probe, eps())))
    values = Float64[]
    for _ in 1:repetitions
        restore!()
        push!(values, cuda_event_time(launch!))
    end
    (; repetitions, times_ms=values, median_ms=median(values),
       mean_ms=mean(values), std_ms=length(values)>1 ? std(values) : 0.0)
end

bandwidth_lower_bound(scalars, milliseconds) =
    8scalars / (milliseconds / 1000) / 1e9

end
