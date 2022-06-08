function small_test()
    
    @testset "Small Allocation Tests" begin

        for i = 1:test_repeat_count

            x = rand(UInt32, 400)
            y = rand(UInt32, 400)

            cu_x = CuArray(x)
            cu_y = CuArray(y)

            ans = QuantumClifford.comm(cu_x, cu_y)
            ref = QuantumClifford.comm(x, y)

            Test.@test ans == ref

        end
        
    end
    
end

function large_test()
    
    @testset "Large Allocation Tests" begin

        for i = 1:test_repeat_count

            x = rand(UInt32, 1024*1024*2*2)
            y = rand(UInt32, 1024*1024*2*2)

            cu_x = CuArray(x)
            cu_y = CuArray(y)

            ans = QuantumClifford.comm(cu_x, cu_y)
            ref = QuantumClifford.comm(x, y)

            Test.@test ans == ref

        end
        
    end
    
end

function weird_count()

    @testset "Weird Allocation Size Tests" begin
    
        for i = 1:test_repeat_count

            x = rand(UInt32, 1023*2)
            y = rand(UInt32, 1023*2)

            cu_x = CuArray(x)
            cu_y = CuArray(y)

            ans = QuantumClifford.comm(cu_x, cu_y)
            ref = QuantumClifford.comm(x, y)

            Test.@test ans == ref

        end
        
    end
    
end

small_test()
large_test()
weird_count()