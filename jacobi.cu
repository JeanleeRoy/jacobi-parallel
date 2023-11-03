#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <time.h>
#include "lib/utils.h"

// Compile: nvcc -o jacobi_cuda jacobi.cu

#define N 100
#define TOL 1.0e-5
#define MAX_ITER 2000

__global__ void jacobi_kernel(double *d_A, double *d_b, double *d_x, double *d_new_x, int* d_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = *d_size;

    if (i < size) {
        d_new_x[i] = d_b[i];
        for (int j = 0; j < size; j++) {
            if (j != i) {
                d_new_x[i] -= d_A[i * size + j] * d_x[j];
            }
        }
        d_new_x[i] /= d_A[i * size + i];
    }
}

void jacobi(double *A, double *b, double *x, int size, int maxIter, double tolerance) { 
    int k = 0;
    double error = tolerance + 1.0;

    int *d_size;
    double *d_A, *d_b, *d_x, *d_new_x;
    double *h_new_x = (double*)calloc(size, size * sizeof(double));

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size * size * sizeof(double));
    cudaMalloc((void**)&d_b, size * sizeof(double));
    cudaMalloc((void**)&d_x, size * sizeof(double));
    cudaMalloc((void**)&d_new_x, size * sizeof(double));
    cudaMalloc((void**)&d_size, sizeof(int));

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
            printf("Converged: %d iterations\n", k + 1);
            break;
        } else {
            cudaMemcpy(d_x, d_new_x, size * sizeof(double), cudaMemcpyDeviceToDevice);
            k++;
        }
    }

    if (k > maxIter) {
        printf("Not converged: %d iterations\n", k);
    }

    printf("Error = %g\n", error);

    // Free allocated memory
    free(h_new_x);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_new_x);
    cudaFree(d_size);
}

int main(int argc, char **argv) {
    int size = N; // Size of the system of equations

    if (argv[1] != NULL) {
        size = atoi(argv[1]);
    }

    double  *A,  // matrix of coefficients 
            *b,  // right vector
            *x;  // initial guess

    A = (double *)malloc(size * size * sizeof(double));
    b = (double *)malloc(size * sizeof(double));
    x = (double *)malloc(size * sizeof(double));

    struct timespec start, end;

    generate_diagonal_dominant_matrix(A, size, 0, 1200);
    generate_vector(b, size, 10, 200);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    jacobi(A, b, x, size, MAX_ITER, TOL);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double delta_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    // Print the solution
    // printf("Solution:\n");
    // for (int i = 0; i < size; i++) {
    //     printf("x[%d] = %.6f\n", i, x[i]);
    // }

    printf("Time elapsed = %g ms\n", delta_ms);

    free(A);
    free(b);
    free(x);

    return 0;
}
