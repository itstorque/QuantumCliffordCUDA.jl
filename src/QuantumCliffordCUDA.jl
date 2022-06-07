"""
A module with GPU kernels for QuantumClifford.jl
"""

module QuantumCliffordCUDA

import QuantumClifford

using CUDA

function test(l::CuArray) where T where L where M
    println("HELLO!")
    println(L)
    println(T)
end

function comm_CUDA(l::CuArray, r::CuArray, out::CuArray)
    # ... complicated CUDA stuff
end

@inline function QuantumClifford.comm(l::CuArray, r::CuArray)::UInt8
    
    println("QuantumClifford.comm thru QuantumCliffordCUDA")

    nthreads = CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    )
    
    nblocks = cld(length(l)/4, nthreads)
    
    out = CuArray([0])
    
    @cuda threads=nthreads blocks=nblocks comm_CUDA(l, r, out)

    CUDA.@allowscalar out[1]

end

end #module