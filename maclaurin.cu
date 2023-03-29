#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


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
    printf("Podaj dokładność N: ");
    scanf("%d", &n);

    threads_per_block = 256;
    blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    sin_x_host = (float *)malloc(sizeof(float));
    sin_x_host[0] = 0.0;
    cudaMalloc(&sin_x_dev, sizeof(float));
    cudaMemcpy(sin_x_dev, sin_x_host, sizeof(float), cudaMemcpyHostToDevice);

    sin_kernel<<<blocks_per_grid, threads_per_block>>>(sin_x_dev, x, n);
    cudaMemcpy(sin_x_host, sin_x_dev, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sin(%lf) = %lf\n", x, *sin_x_host);

    free(sin_x_host);
    cudaFree(sin_x_dev);

    return 0;
}

