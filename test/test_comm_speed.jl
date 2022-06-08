function large_test()
    
    @testset "Large Speed Tests" begin

        x = rand(UInt32, 1024*1024*2*2)
        y = rand(UInt32, 1024*1024*2*2)

        cu_x = CuArray(x)
        cu_y = CuArray(y)

        bench     = @benchmark CUDA.@sync QuantumClifford.comm($cu_x, $cu_y)
        bench_ref = @benchmark QuantumClifford.comm($x, $y)
        
        time_ratio = ratio(minimum(bench_ref), minimum(bench)).time

        Test.@test time_ratio > 2 # fail if less than a 2 times speed up
        
        Test.@test time_ratio > 5 # fail if less than a 5 times speed up
        
        Test.@test time_ratio > 10 # fail if less than a 5 times speed up
        
        println("Speed-Up:", time_ratio)
        
    end
    
end

large_test()