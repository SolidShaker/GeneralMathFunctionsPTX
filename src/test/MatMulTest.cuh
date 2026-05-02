#ifndef MATMULTEST_H
#define MATMULTEST_H

#include <cuda_f16.h>
#include <cuda_runtime.h>
#include <iostream>




namespace TEST
{
    __global__ void MatMul(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
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
