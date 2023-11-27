#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/utils.h"

// compile: nvc -acc -o jacobi_acc jacobi-openacc_v2.c

#define N 100
#define MAX_ITER 500
#define TOLERANCE 1e-5

void jacobi(double *A, double *b, double *x, int size, int maxIter, double tolerance) {
    int k = 0;
    double error = tolerance + 1.0;

    double *new_x = (double *)malloc(size * sizeof(double));

    // Allocate and copy A, b, x and allocate new_x on the GPU
    #pragma acc data copyin(A[0 : size * size], b[0 : size], x[0 : size]) create(new_x[0 : size])
    {
        while (k < maxIter && error > tolerance) {
            #pragma acc parallel loop
            for (int i = 0; i < size; i++) {
                new_x[i] = b[i];
                for (int j = 0; j < size; j++) {
                    if (j != i) {
                        new_x[i] -= A[i * size + j] * x[j];
                    }
                }
                if (A[i * size + i] != 0) {
                    new_x[i] /= A[i * size + i];
                } else {
                    new_x[i] = 0;
                }
            }

            // Copy results back from the device to the host
            #pragma acc update self(new_x[0:size])

            error = 0.0;
            for (int i = 0; i < size; i++) {
                error += fabs(new_x[i] - x[i]);
            }

            for (int i = 0; i < size; i++) {
                x[i] = new_x[i];
            }

            // copy x to the device
            #pragma acc update device(x[0 : size])

            k++;
        }
    }

    if (error < tolerance) {
        printf("Converged: %d iterations\n", k);
    } else {
        printf("Not converged: %d iterations\n", k);
    }

    printf("Error = %g\n", error);

    // Deallocate the GPU memory
    #pragma acc exit data delete(A, b, x, new_x)
}

int main(int argc, char **argv) {
    int size;  // Size of the system of equations

    double *A,  // diagonal matrix of coefficients
        *b,     // right vector
        *x;     // initial guess

    Data data = init_data(argc, argv, N);

    size = data.size;
    A = data.A;
    b = data.b;
    x = data.x;

    print_header("Jacobi method");

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    jacobi(A, b, x, size, MAX_ITER, TOLERANCE);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double delta_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    //  Print the solution
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

// Analyse:

// perf record ./jacobi_acc
// perf report'

// sudo perf stat ./jacobi_acc
