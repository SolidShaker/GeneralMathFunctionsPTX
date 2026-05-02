#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <cuda_fp16.h>



namespace FP
{
    template<int BLOCK_SIZE>
    __global__ void MatMul(
            const half* __restrict__ A,
            const half* __restrict__ B,
            half* __restrict__ C,
            int M, int N, int K)
    {
        __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];

        int by = blockIdx.y,  bx = blockIdx.x;
        int ty = threadIdx.x, tx = threadIdx.x;

        int row = by * BLOCK_SIZE + ty;
        int col = bx * BLOCK_SIZE + tx;

        float sum = 0.f;

        for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile)
        {
            if (row < M && tile * BLOCK_SIZE + tx < K) 
                As[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
            else 
                As[ty][tx] = __float2half(0.f);

            if (tile * BLOCK_SIZE + ty < K && col < N)
                Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
            else
                Bs[ty][tx] = __float2half(0.f);
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k)
                sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
            __syncthreads();
        }

        if (row < M && col < N)
            C[row * N + col] = __float2half(sum);
    }
};


#endif
