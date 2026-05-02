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
        float maxMismatch = 0.0f;
        
        for (int i = 0; i < M * N; i++) 
        {
            float val1 = __half2float(first[i]);
            float val2 = __half2float(second[i]);
            float difference = std::abs(val1 - val2);  // FIXED: use float, not int
            
            if (difference > tolerance) 
            {
                totalMismatch += difference;
                counterMismatch++;
                maxMismatch = std::max(maxMismatch, difference);
            }
        }
        
        float mismatchRate = (counterMismatch / (float)(M * N)) * 100.0f;
        
        std::cout << "Verification Results:\n";
        std::cout << "  Total mismatches: " << counterMismatch << " / " << (M * N) 
                  << " (" << mismatchRate << "%)\n";
        std::cout << "  Total error sum: " << totalMismatch << "\n";
        std::cout << "  Maximum error: " << maxMismatch << "\n";
        std::cout << "  Average error: " << (counterMismatch > 0 ? totalMismatch / counterMismatch : 0) << "\n";
        
        if (counterMismatch == 0) {
            std::cout << "  Result: ✓ PASSED (all within tolerance)\n";
        } else {
            std::cout << "  Result: ✗ FAILED (" << counterMismatch << " elements outside tolerance)\n";
        }
    }
}


#endif
