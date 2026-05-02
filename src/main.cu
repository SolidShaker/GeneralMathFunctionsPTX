#include "MatMul.cuh"
#include "test/MatMulTest.cuh"

__host__ __forceinline__ int GetPadding(int size, int delimeter)
{
    return size % delimeter == 0 ? size : 
            (size + delimeter-1) & ~(delimeter-1);
}


int main()
{
    const int M = GetPadding(512, 16);
    const int N = GetPadding(512, 16);
    const int K = GetPadding(512, 16);

    half* hA = new half[M * K];
    half* hB = new half[K * N];
    half* hC1 = new half[M * N];
    half* hC2 = new half[M * N];

    for (int i = 0; i < M * K; i++) hA[i] = __float2half(1.f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2half(2.f);   


    half* dA;
    half* dB;
    half* dC1;
    half* dC2;

    cudaMalloc(&dA, M * K * sizeof(half));
    cudaMalloc(&dB, K * N * sizeof(half));
    cudaMalloc(&dC1, M * N * sizeof(half));
    cudaMalloc(&dC2, M * N * sizeof(half));

    cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    FP::MatMul<16><<<blocks, threads>>>(dA, dB, dC1, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(hC1, dC1, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    TEST::MatMul<<<blocks, threads>>>(dA, dB, dC2, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(hC2, dC2, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    // TEST::VerifyResult(hC1, hC2, M, N);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC1);
    cudaFree(dC2);
    return 0;
}
