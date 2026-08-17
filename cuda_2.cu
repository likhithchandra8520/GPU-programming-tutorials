#include <iostream>
#include <cuda_runtime.h>
__global__ void barrierKernel(int *counter){
    int tid = threadIdx.x;
    printf("Thread %d reached the barrier\n", tid);
    atomicAdd(counter,1);
    __syncthreads();
    if (tid == 0){
        printf("\nNumber of threads reached barrier: %d\n", *counter);
        if (*counter == blockDim.x) printf("All threads have reached the barrier!\n");
    }
    __syncthreads();
    printf("Thread %d passed the barrier\n", tid);
}

int main(){
    int numThreads = 8;
    int *d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    barrierKernel<<<1, numThreads>>>(d_counter);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << std::endl;
    }
    cudaFree(d_counter);
    return 0;
}