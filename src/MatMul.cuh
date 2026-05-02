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

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Global row and column for this thread
        int row = by * BLOCK_SIZE + ty;
        int col = bx * BLOCK_SIZE + tx;

        float sum = 0.0f;

        // Loop over tiles
        for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) 
        {
            // Load tile from A into shared memory
            int a_row = row;
            int a_col = tile * BLOCK_SIZE + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[a_row * K + a_col];
            } else {
                As[ty][tx] = __float2half(0.0f);
            }

            // Load tile from B into shared memory
            int b_row = tile * BLOCK_SIZE + ty;
            int b_col = col;
            if (b_row < K && col < N) {
                Bs[ty][tx] = B[b_row * N + b_col];
            } else {
                Bs[ty][tx] = __float2half(0.0f);
            }

            __syncthreads();

            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
            }

            __syncthreads();
        }

        // Write result
        if (row < M && col < N) {
            C[row * N + col] = __float2half(sum);
        }
    }
};


#endif
