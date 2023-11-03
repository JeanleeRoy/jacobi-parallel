#ifndef GENERATE_MATRIX_H
#define GENERATE_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void generate_diagonal_matrix(double *matrix, int size, int min, int max) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        matrix[i * size + i] = (double)(rand() % (max - min + 1) + min);
    }
}

void generate_diagonal_dominant_matrix(double *matrix, int size, int min, int max) {
    srand(time(NULL)); // random seed

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                // Diagonal element, assign a random value within the range [min, max]
                matrix[i * size + j] = (rand() % (max - min + 1)) + min;
            } else {
                // Non-diagonal element, assigns a random value close to zero
                double value = ((rand() % 1000) / 1000.0) * (min > 0 ? min : 1);
                matrix[i * size + j] = value < 0.95 ? 0 : value;
            }
        }
    }

    // Make sure the matrix is ​​diagonal dominant
    for (int i = 0; i < size; i++) {
        double sum = 0.0;
        for (int j = 0; j < size; j++) {
            if (i != j) {
                sum += abs(matrix[i * size + j]);
            }
        }
        matrix[i * size + i] += sum;
    }
}

void generate_vector(double *matrix, int size, int min, int max) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (double)(rand() % (max - min + 1) + min);
    }
}

void print_matrix(double* A, int n, int m) {
    for (int i = 0; i < n; i++) {
        printf("[");
        for (int j = 0; j < m; j++) {
            printf("%.2f\t", A[i * m + j]);
        }
        printf("]\n");
    }
}

void print_vector(double* A, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.2f\t", A[i]);
    }
    printf("]\n");
}

#endif  // GENERATE_MATRIX_H
