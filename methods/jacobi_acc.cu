#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// compile: nvc -acc -c jacobi_acc.cu -lm

extern "C" void jacobi_acc(double *A, double *b, double *x, int size, int maxIter, double tolerance) {
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
                        new_x[i] -= A[j * size + i] * x[j];
                    }
                }
                if (A[i * size + i] != 0) {
                    new_x[i] /= A[i * size + i];
                } else {
                    new_x[i] = 0;
                }
            }

            // Copy results back from the device to the host
            #pragma acc update self(new_x[0 : size])

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
        printf(" Converged: \t  %d iterations\n", k);
    } else {
        printf(" Not converged: \t %d iterations\n", k);
    }

    printf(" Error: \t  %g\n", error);

    // Deallocate the GPU memory
    #pragma acc exit data delete (A, b, x, new_x)

    // Free the host memory
    free(new_x);
}
