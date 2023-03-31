#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>


__global__ void sin_kernel(float *sin_x, float x, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        int sign = (i % 2 == 0) ? 1 : -1;
        
	float power = 1.0;
	for (int j = 0; j < 2 * i + 1; j++) power *= x;
	int factorial = 1;
	for (int j = 1; j <= 2 * i + 1; j++) factorial *= j;

        float term = sign * power / factorial;
	atomicAdd(sin_x, term);
    }
}

int main()
{
    float x, *sin_x_host, *sin_x_dev;
    int n, threads_per_block, blocks_per_grid;

    printf("Podaj x w radianach: ");
    scanf("%f", &x);
    n = 15;

    threads_per_block = 256;
    blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    clock_t start, end, computing_start, computing_end;
    double cpu_time_used;

    start = clock();
    
    sin_x_host = (float *)malloc(sizeof(float));
    sin_x_host[0] = 0.0;
    cudaMalloc(&sin_x_dev, sizeof(float));
    cudaMemcpy(sin_x_dev, sin_x_host, sizeof(float), cudaMemcpyHostToDevice);
    
    computing_start = clock();
    sin_kernel<<<blocks_per_grid, threads_per_block>>>(sin_x_dev, x, n);
    computing_end = clock();
    cudaMemcpy(sin_x_host, sin_x_dev, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sin(%lf) = %lf\n", x, *sin_x_host);

    free(sin_x_host);
    cudaFree(sin_x_dev);

    end = clock();
    cpu_time_used = ((double) (end - start)/CLOCKS_PER_SEC);
    printf("Total time elapsed: %lf\n", cpu_time_used);
    cpu_time_used = ((double) (computing_end - computing_start)/CLOCKS_PER_SEC);
    printf("Computing time elapsed: %lf\n", cpu_time_used);

    return 0;
}

