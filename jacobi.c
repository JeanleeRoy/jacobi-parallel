#include <stdio.h>
#include <math.h>
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
    int size = N; // Size of the system of equations

    if (argv[1] != NULL) {
        size = atoi(argv[1]);
    }

    double  *A, // diagonal matrix of coefficients 
            *b, // right vector
            *x; // Initial guess
    
    A = (double *)malloc(size * size * sizeof(double));
    b = (double *)malloc(size * sizeof(double));
    x = (double *)malloc(size * sizeof(double));

    struct timespec start, end;

    generate_diagonal_dominant_matrix(A, size, 0, 1200);
    generate_vector(b, size, 10, 200);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    jacobi(A, b, x, size, TOL, MAX_ITER);

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

    // double A[N][N] = {{5, 1, 1}, {1, 6, 2}, {2, 3, 7}}; // Coefficient 
    // double b[N] = {10, 15, 20}; // right vector
    // double x[N] = {0}; // Initial guess
