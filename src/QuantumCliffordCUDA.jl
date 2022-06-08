"""
A module with GPU kernels for QuantumClifford.jl
"""

module QuantumCliffordCUDA

import QuantumClifford

using CUDA

# grid atomic implementation
function comm_CUDA(l::AbstractArray, r::AbstractArray, out::AbstractArray)
    
    elements = blockDim().x*2
    thread = threadIdx().x
    block = blockIdx().x
    offset = (block-1) * elements
    
    len = length(l) >> 1

    # parallel reduction of values in a block
    d = 1
    while d < elements
        sync_threads()
        index = 2 * d * (thread-1) + 1
        @inbounds if index <= elements && offset+index+d <= len
            
            if (d==1)
                
                if (offset+index+d == len+1)
                    
                    l[offset+index] = ((l[offset+index+len] & r[offset+index]) ⊻ (l[offset+index] & r[offset+index+len]))
                    
                else
                
                    l[offset+index] = ((l[offset+index+len] & r[offset+index]) ⊻ (l[offset+index] & r[offset+index+len])) ⊻
                                      ((l[offset+index+len+d] & r[offset+index+d]) ⊻ (l[offset+index+d] & r[offset+index+len+d]))
                    
                end
                
            else
                
                if (offset+index+d >= len)
                    
                else

                    l[offset+index] = l[offset+index] ⊻ l[offset+index+d]
                    
                end
                
            end
            
        end
        
        d *= 2
    end
    
    # atomic reduction of this block's value
    if thread == 1
        CUDA.@atomic out[] = out[] ⊻ l[offset + 1]
    end
    
    return
    
end

@inline function QuantumClifford.comm(l::CuArray, r::CuArray)::UInt8

    nthreads = CUDA.attribute(
        device(),
        CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    )
    
    nblocks = cld(length(l), 4*nthreads)
    
    out = CuArray([0])
    
    @cuda threads=nthreads blocks=nblocks comm_CUDA(l, r, out)

    # convert(UInt8, count_ones(CUDA.@allowscalar out[1])) % 4
    
    count_ones(CUDA.@allowscalar out[1]) % 2

end

end #module