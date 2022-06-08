using CUDA
using QuantumClifford
using QuantumCliffordCUDA

using Test

function small_test()
    
    println("small_test")

    for i = 1:10

        x = rand(UInt32, 400)
        y = rand(UInt32, 400)

        cu_x = CuArray(x)
        cu_y = CuArray(y)

        ans = QuantumClifford.comm(cu_x, cu_y)
        ref = QuantumClifford.comm(x, y)

        Test.@test ans == ref

    end
    
end

function large_test()
    
    println("large_test")

    for i = 1:10

        x = rand(UInt32, 1024*1024*2*2)
        y = rand(UInt32, 1024*1024*2*2)

        cu_x = CuArray(x)
        cu_y = CuArray(y)

        ans = QuantumClifford.comm(cu_x, cu_y)
        ref = QuantumClifford.comm(x, y)

        Test.@test ans == ref

    end
    
end

function weird_count()
    
    println("weird_count")

    for i = 1:10

        x = rand(UInt32, 1023*2)
        y = rand(UInt32, 1023*2)

        cu_x = CuArray(x)
        cu_y = CuArray(y)

        ans = QuantumClifford.comm(cu_x, cu_y)
        ref = QuantumClifford.comm(x, y)

        Test.@test ans == ref

    end
    
end

small_test()
large_test()
weird_count()