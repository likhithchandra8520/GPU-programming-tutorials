#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace wmma;

#define N 64
#define TILE 16

__global__ void initializeMatrices(half *A, half *B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N)
    {
        A[idx] = __float2half(1.0f);
        B[idx] = __float2half(1.0f);
    }
}

__global__ void tensorCoreMatMul(half *A, half *B, float *C)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadId / 32;
    if (warpId >= 16)
        return;

    int tileRow = warpId / 4;
    int tileCol = warpId % 4;
    int row = tileRow * TILE;
    int col = tileCol * TILE;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < 4; k++)
    {
        int A_index = row * N + k * TILE;
        int B_index = k * TILE * N + col;

        load_matrix_sync(a_frag, A + A_index, N);
        load_matrix_sync(b_frag, B + B_index, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int C_index = row * N + col;
    store_matrix_sync(C + C_index, c_frag, N, mem_row_major);
}

int main()
{
    half *A;
    half *B;
    float *C;
    float h_C[N * N];

    cudaMalloc(&A, N * N * sizeof(half));
    cudaMalloc(&B, N * N * sizeof(half));
    cudaMalloc(&C, N * N * sizeof(float));

    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;

    initializeMatrices<<<blocks, threads>>>(A, B);
    cudaDeviceSynchronize();

    tensorCoreMatMul<<<1, 512>>>(A, B, C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (h_C[i * N + j] != 64.0f)
            {
                correct = false;
                printf("Error at C[%d][%d] = %f\n", i, j, h_C[i * N + j]);
                break;
            }
        }
        if (!correct)
            break;
    }

    if (correct)
        printf("Matrix multiplication successful!\n64x64 matrices multiplied using 16x16 Tensor Core tiles.\n");
    else
        printf("Matrix multiplication failed!\n");

    printf("\nFirst 4x4 elements of C:\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            printf("%6.1f ", h_C[i * N + j]);
        printf("\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
```