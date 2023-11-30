#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Compile: nvcc -c jacobi_cuda.cu -lm

__global__ void jacobi_kernel(double *d_A, double *d_b, double *d_x, double *d_new_x, int *d_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = *d_size;

    if (i < size) {
        d_new_x[i] = d_b[i];
        for (int j = 0; j < size; j++) {
            if (j != i) {
                d_new_x[i] -= d_A[j * size + i] * d_x[j];
            }
        }

        if (d_A[i * size + i] != 0) {
            d_new_x[i] /= d_A[i * size + i];
        } else {
            d_new_x[i] = 0;
        }
    }
}

extern "C" void jacobi_cuda(double *A, double *b, double *x, int size, int maxIter, double tolerance) {
    int k = 0;
    double error = tolerance + 1.0;

    int *d_size;
    double *d_A, *d_b, *d_x, *d_new_x;
    double *h_new_x = (double *)calloc(size, size * sizeof(double));

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size * size * sizeof(double));
    cudaMalloc((void **)&d_b, size * sizeof(double));
    cudaMalloc((void **)&d_x, size * sizeof(double));
    cudaMalloc((void **)&d_new_x, size * sizeof(double));
    cudaMalloc((void **)&d_size, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);

    // Kernel GPU dimensions
    int block_size = 512;
    int num_blocks = (size + block_size - 1) / block_size;

    while (k < maxIter && error > tolerance) {
        jacobi_kernel<<<num_blocks, block_size>>>(d_A, d_b, d_x, d_new_x, d_size);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }

        cudaMemcpy(h_new_x, d_new_x, size * sizeof(double), cudaMemcpyDeviceToHost);

        error = 0.0;
        for (int i = 0; i < size; i++) {
            error += fabs(h_new_x[i] - x[i]);
        }

        for (int i = 0; i < size; i++) {
            x[i] = h_new_x[i];
        }

        if (error < tolerance) {
            printf(" Converged: \t  %d iterations\n", k + 1);
            break;
        } else {
            cudaMemcpy(d_x, d_new_x, size * sizeof(double), cudaMemcpyDeviceToDevice);
            k++;
        }
    }

    if (k > maxIter) {
        printf(" Not converged: %d iterations\n", k);
    }

    printf(" Error: \t  %g\n", error);

    // Free allocated memory
    free(h_new_x);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_new_x);
    cudaFree(d_size);
}