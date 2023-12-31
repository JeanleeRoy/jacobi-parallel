#include <math.h>
#include <stdio.h>

#include "lib/utils.h"

// Compile: gcc -g -o jacobi jacobi.c

#define N 100
#define TOL 1.0e-5
#define MAX_ITER 2000

// Jacobi method
void jacobi(double *A, double *b, double *x, int size, double tolerance, int maxIter) {
    int i, j, k = 0;
    double error = tolerance + 1.0;

    double *new_x = (double *)calloc(size, size * sizeof(double));

    while (k < maxIter && error > tolerance) {
        for (i = 0; i < size; i++) {
            new_x[i] = b[i];
            for (j = 0; j < size; j++) {
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

        // Calculate the error
        error = 0.0;
        for (i = 0; i < size; i++) {
            error += fabs(new_x[i] - x[i]);
        }

        // Update the solution
        for (i = 0; i < size; i++) {
            x[i] = new_x[i];
        }

        k++;
    }

    if (error <= tolerance) {
        printf("Converged: %d iterations\n", k);
    } else {
        printf("Did not converge after %d iterations\n", maxIter);
    }

    printf("Error = %g\n", error);

    free(new_x);
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

    jacobi(A, b, x, size, TOL, MAX_ITER);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double delta_ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

    printf("Time elapsed = %g ms\n\n", delta_ms);
    
    // Print the solution
    // printf("Solution [x]:\n");
    // print_vector_inline(x, size);
    // for (int i = 0; i < size; i++) {
    //     printf("x[%d] = %.6f\n", i, x[i]);
    // }


    free(A);
    free(b);
    free(x);

    return 0;
}

// double A[N][N] = {{5, 1, 1}, {1, 6, 2}, {2, 3, 7}}; // Coefficient
// double b[N] = {10, 15, 20}; // right vector
// double x[N] = {0}; // Initial guess
