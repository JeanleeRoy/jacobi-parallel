#include <stdio.h>
#include <math.h>
#include <stdlib.h> 
#include <openacc.h>

// Obsolete file

#include "lib/utils.h"

#define N 10
#define MAX_ITER 100
#define TOLERANCE 1e-6

void jacobi(double *A, double *b, double *x, int size, double tolerance, int max_iter) {
    int i, j, k;
    double sum;
    double *new_x = (double *)malloc(size * sizeof(double));

    for (k = 0; k < max_iter; k++) {
        #pragma acc parallel loop private(sum)
        for (i = 0; i < size; i++) {
            sum = 0.0;
            for (j = 0; j < size; j++) {
                if (i != j) {
                    sum += A[i * size + j] * x[j];
                }
            }
            if (A[i * size + i] != 0) {
                new_x[i] = (b[i] - sum) / A[i * size + i];
            } else {
                new_x[i] = 0;
            }
        }

        int converged = 1;
        #pragma acc parallel loop reduction(&&:converged)
        for (i = 0; i < size; i++) {
            if (fabs(x[i] - new_x[i]) > tolerance) {
                converged = 0;
            }
        }

        if (converged) {
            printf("Converged after %d iterations\n", k + 1);
            break;
        }

        #pragma acc parallel loop
        for (i = 0; i < size; i++) {
            x[i] = new_x[i];
        }
    }

    free(new_x);
}

int main(int argc, char **argv) {
    const int custom_size = atoi(argv[1]);
    const int size = N; // Size of the system of equations

    // // double A[N * N] = {5, 1, 1,  1, 6, 2,  2, 3, 7}; // Coefficients
    // double A[N * N] = {0};
    // generate_diagonal_matrix(A, N, 0, 100);
    // double b[N] = {100, 150, 200}; // right vector
    // double x[N] = {0}; // Initial guess

    double  *A,  // matrix of coefficients 
            *b,  // right vector
            *x;  // initial guess
    
    A = (double *)malloc(size * size * sizeof(double));
    b = (double *)malloc(size * sizeof(double));
    x = (double *)malloc(size * sizeof(double));

    generate_diagonal_dominant_matrix(A, size, 0, 100);
    generate_vector(b, size, 50, 500);
    // generate_vector(x, size, 0, 100)

    // print_matrix(A, size, size);

    jacobi(A, b, x, size, TOLERANCE, MAX_ITER);

     // Print the solution
    printf("Solution:\n");
    for (int i = 0; i < size; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    free(A);
    free(b);
    free(x);

    return 0;
}
