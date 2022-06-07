using CUDA
using QuantumClifford
using QuantumCliffordCUDA

x = rand(UInt32, 1024*6*2*2)
y = rand(UInt32, 1024*6*2*2)

cu_x = CuArray(x)
cu_y = CuArray(y)

ans = QuantumClifford.comm(cu_x, cu_y)

println(ans)