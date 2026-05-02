#ifndef MATMULTEST_H
#define MATMULTEST_H

#include <cuda_runtime.h>
#include <iostream>




namespace TEST
{
    __global__ void MatMul(
        const half* __restrict__ A,
        const half* __restrict__ B,
        half* __restrict__ C,
        int M, int N, int K)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
            }
            C[row * N + col] = __float2half(sum);
        }
    }

    void
    VerifyResult(half* first, half* second, int M, int N, float tolerance = 0.01f) 
    {
        float totalMismatch = 0.f;
        int counterMismatch = 0;
        for (int i = 0; i < M * N; i++) 
        {
            int difference = std::abs(__half2float(first[i]) - __half2float(second[i]));
            if (difference > tolerance) 
            {
                totalMismatch += difference;
                counterMismatch++;
            }
        }
        std::cout << "TotalMismatch" << totalMismatch << "CounterMismatch" << counterMismatch << std::endl;        
    }
}


#endif
