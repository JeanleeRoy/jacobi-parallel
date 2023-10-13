#include <stdio.h>
#include <math.h>
#include <stdlib.h> 

#define N 3
#define MAX_ITER 1000
#define TOLERANCE 1e-6

void jacobi(double A[], double b[], double x[]) {
    int i, j, k;
    double sum;
    double *new_x = (double *)malloc(N * sizeof(double));

    for (k = 0; k < MAX_ITER; k++) {
        #pragma acc parallel loop private(sum)
        for (i = 0; i < N; i++) {
            sum = 0.0;
            for (j = 0; j < N; j++) {
                if (i != j) {
                    sum += A[i * N + j] * x[j];
                }
            }
            new_x[i] = (b[i] - sum) / A[i * N + i];
        }

        int converged = 1;
        #pragma acc parallel loop reduction(&&:converged)
        for (i = 0; i < N; i++) {
            if (fabs(x[i] - new_x[i]) > TOLERANCE) {
                converged = 0;
            }
        }

        if (converged) {
            printf("Converged after %d iterations\n", k + 1);
            break;
        }

        #pragma acc parallel loop
        for (i = 0; i < N; i++) {
            x[i] = new_x[i];
        }
    }

    free(new_x);
}

int main() {
    double A[N * N] = {5, 1, 1,  1, 6, 2,  2, 3, 7}; // Coefficients
    double b[N] = {10, 15, 20}; // right vector
    double x[N] = {0}; // Initial guess

    jacobi(A, b, x);

     // Print the solution
    printf("Solution:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    return 0;
}

// compilar: nvc -acc g -o jacobi_acc jacobi-openacc.c 

// Analyse:

// perf record ./jacobi_acc
// perf report'

// sudo perf stat ./jacobi_acc