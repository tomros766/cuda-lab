#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <time.h>


__global__ void trapezoidal_rule(float* d_result, float a, float b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float h = (b - a) / n;
    int num_threads = gridDim.x * blockDim.x;
    int chunk_size = n / num_threads;
    int start = index * chunk_size;
    int end = (index == num_threads-1) ? n : start + chunk_size;

    float x0 = a + start * h;
    float xn = a + end * h;
    float f0 = 4/(1 + x0 * x0);
    float fn = 4/(1 + xn * xn);
    float s = (f0 + fn) / 2.0;
    for (int i = start + 1; i < end; i++) {
        float xi = a + i * h;
	float fi = 4/(1 + xi * xi);
        s += fi;
    }
    atomicAdd(d_result, s * h);
}

int main() {
    float a = 0, b = 1;
    int n = 128;
    int num_threads = 1;
    int num_blocks = (n + num_threads - 1) / num_threads;


    clock_t start, end, computing_start, computing_end;
    float cpu_time_used;
    start = clock();

    float* h_result = new float[0];
    h_result[0] = 0.0;
    float* d_result;

    cudaMalloc((void**)&d_result,  sizeof(float));
    cudaMemcpy(d_result, h_result,  sizeof(float), cudaMemcpyHostToDevice);

    computing_start = clock();
    trapezoidal_rule<<<num_blocks, num_threads>>>(d_result, a, b, n);
    computing_end = clock();

    cudaMemcpy(h_result, d_result,  sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Estimated: " << *h_result << std::endl;
    delete[] h_result;
    cudaFree(d_result);

    end = clock();
    cpu_time_used = ((double) (end - start)/CLOCKS_PER_SEC);
    printf("Total time elapsed: %lf\n", cpu_time_used);
    cpu_time_used = ((double) (computing_end - computing_start)/CLOCKS_PER_SEC);
    printf("Computing time elapsed: %lf\n", cpu_time_used);

    return 0;
}

