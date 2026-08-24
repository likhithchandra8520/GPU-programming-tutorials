#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#define INF INT_MAX

struct Edge {
    int src;
    int dst;
    int weight;
};

__global__ void ssspKernel(const int *rowPtr, const int *colIdx,
                           const int *weights, int *dist,
                           int *changed, int V) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;

    if (u >= V)
        return;

    if (dist[u] == INF)
        return;

    for (int e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
        int v = colIdx[e];
        int w = weights[e];
        int newDist = dist[u] + w;

        if (newDist < dist[v]) {
            atomicMin(&dist[v], newDist);
            *changed = 1;
        }
    }
}

int main() {
    std::ifstream file("graph.txt");

    if (!file) {
        std::cerr << "Error: Could not open graph.txt\n";
        return 1;
    }

    int V, E;
    file >> V >> E;

    std::vector<Edge> edges(E);

    for (int i = 0; i < E; i++) {
        file >> edges[i].src >> edges[i].dst >> edges[i].weight;
    }

    file.close();

    std::vector<int> rowPtr(V + 1, 0);

    for (const auto &edge : edges) {
        rowPtr[edge.src + 1]++;
    }

    for (int i = 1; i <= V; i++) {
        rowPtr[i] += rowPtr[i - 1];
    }

    std::vector<int> colIdx(E);
    std::vector<int> weights(E);
    std::vector<int> position = rowPtr;

    for (const auto &edge : edges) {
        int index = position[edge.src]++;
        colIdx[index] = edge.dst;
        weights[index] = edge.weight;
    }

    std::cout << "CSR Representation\n";

    std::cout << "rowPtr: ";
    for (int x : rowPtr)
        std::cout << x << " ";
    std::cout << "\n";

    std::cout << "colIdx: ";
    for (int x : colIdx)
        std::cout << x << " ";
    std::cout << "\n";

    std::cout << "weights: ";
    for (int x : weights)
        std::cout << x << " ";
    std::cout << "\n\n";

    int *d_rowPtr, *d_colIdx, *d_weights, *d_dist, *d_changed;

    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_rowPtr, rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), E * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> dist(V, INF);
    dist[0] = 0;

    cudaMemcpy(d_dist, dist.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (V + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < V - 1; i++) {
        int changed = 0;

        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        ssspKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_rowPtr, d_colIdx, d_weights, d_dist, d_changed, V
        );

        cudaDeviceSynchronize();

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        if (!changed)
            break;
    }

    cudaMemcpy(dist.data(), d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Shortest distances from source vertex 0:\n";

    for (int i = 0; i < V; i++) {
        if (dist[i] == INF)
            std::cout << "Vertex " << i << ": INF\n";
        else
            std::cout << "Vertex " << i << ": " << dist[i] << "\n";
    }

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_changed);

    return 0;
}