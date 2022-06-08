using CUDA
using QuantumClifford
using QuantumCliffordCUDA

LARGE_COMPUTATION = 1024*1024*2*2

for i = 1:10

    x = rand(UInt32, 400)
    y = rand(UInt32, 400)

    cu_x = CuArray(x)
    cu_y = CuArray(y)

    ans = QuantumClifford.comm(cu_x, cu_y)
    ref = QuantumClifford.comm(x, y)

    print("CUDA:")
    println(ans)

    print("REF: ")
    println(ref)
    
end