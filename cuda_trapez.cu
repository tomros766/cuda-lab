#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void trapezoidal_rule(double* d_result, double a, double b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double h = (b - a) / n;
    int num_threads = gridDim.x * blockDim.x;
    int chunk_size = n / num_threads;
    int start = index * chunk_size;
    int end = (index == num_threads-1) ? n : start + chunk_size;

    double x0 = a + start * h;
    double xn = a + end * h;
    double f0 = 4/(1 + x0 * x0);
    double fn = 4/(1 + xn * xn);
    double s = (f0 + fn) / 2.0;
    for (int i = start + 1; i < end; i++) {
        double xi = a + i * h;
	double fi = 4/(1 + xi * xi);
        s += fi;
    }
    d_result[index] = s * h;
}

int main() {
    double a = 0, b = 1;
    int n = 128;
    int num_threads = 1;
    int num_blocks = (n + num_threads - 1) / num_threads;

double* h_result = new double[num_blocks];
    h_result[0] = 0.0;
    double* d_result;

    cudaMalloc((void**)&d_result, num_blocks * sizeof(double));
    cudaMemcpy(d_result, h_result, num_blocks * sizeof(double), cudaMemcpyHostToDevice);

        trapezoidal_rule<<<num_blocks, num_threads>>>(d_result, a, b, n);

    cudaMemcpy(h_result, d_result, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < num_blocks; i++) {
	sum += h_result[i];    
    }
    std::cout << "Estimated: " << sum << std::endl;
    delete[] h_result;
    cudaFree(d_result);
    return 0;
}

