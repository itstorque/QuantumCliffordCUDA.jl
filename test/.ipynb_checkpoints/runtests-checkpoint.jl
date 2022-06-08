using CUDA
using QuantumClifford
using QuantumCliffordCUDA

using Test

test_sizes = [] # incorporate this later...
test_repeat_count = 10

function doset(descr)
    if length(ARGS) == 0
        return true
    end
    for a in ARGS
        if occursin(lowercase(a), lowercase(descr))
            return true
        end
    end
    return false
end

doset("gpu_comm")             && include("./test_comm.jl")