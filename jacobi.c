#include <stdio.h>
#include <math.h>

// Max iterations
#define MAX_ITER 1000
#define MAX_SIZE 100

// Jacobi method
void jacobi(double A[][MAX_SIZE], double b[], double x[], int n, double tolerance) {
    double new_x[MAX_SIZE];
    int i, j, iter;
    
    for (iter = 0; iter < MAX_ITER; iter++) {
        for (i = 0; i < n; i++) {
            new_x[i] = b[i];
            for (j = 0; j < n; j++) {
                if (j != i) {
                    new_x[i] -= A[i][j] * x[j];
                }
            }
            new_x[i] /= A[i][i];
        }

        // Calculate the error
        double error = 0.0;
        for (i = 0; i < n; i++) {
            error += fabs(new_x[i] - x[i]);
        }

        // Update the solution
        for (i = 0; i < n; i++) {
            x[i] = new_x[i];
        }

        // Check for convergence
        if (error < tolerance) {
            printf("Converged after %d iterations\n", iter + 1);
            return;
        }
    }
    
    printf("Did not converge after %d iterations\n", MAX_ITER);
}

int main() {
    int n = 3; // Size of the system of equations
    double A[MAX_SIZE][MAX_SIZE] = {{5, 1, 1}, {1, 6, 2}, {2, 3, 7}}; // Coefficient 
    double b[MAX_SIZE] = {10, 15, 20}; // right vector
    double x[MAX_SIZE] = {0}; // Initial guess

    double tolerance = 1e-6; // Tolerance for convergence

    jacobi(A, b, x, n, tolerance);

    // Print the solution
    printf("Solution:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    return 0;
}

// Compilar: cc -g -o jacobi jacobi.c
