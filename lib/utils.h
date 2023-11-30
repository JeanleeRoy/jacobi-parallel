#ifndef GENERATE_MATRIX_H
#define GENERATE_MATRIX_H

#include <math.h>
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
    srand(time(NULL));  // random seed

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

void print_matrix(double *A, int n, int m) {
    for (int i = 0; i < n; i++) {
        printf("[");
        for (int j = 0; j < m; j++) {
            printf("%.2f\t", A[i * n + j]);
        }
        printf("]\n");
    }
}

void print_array_inline(double *a, int n, int is_fortran) {
    if (is_fortran) {
        int size = (int)sqrt(n);
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                printf("%.6f\t", a[i * size + j]);
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            printf("%.6f\t", a[i]);
        }
    }
    printf("\n");
}

void print_vector_inline(double *v, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.6f\t", v[i]);
    }
    printf("\n");
}

void print_vector(double *A, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.2f\t", A[i]);
    }
    printf("]\n");
}

void print_header(char *title) {
    printf("\n");
    printf("========================================\n");
    printf("%s\n", title);
    printf("========================================\n");
}

typedef struct {
    int size;
    double *A;
    double *b;
    double *x;
} Data;

Data read_data_from_file(char *filename, int verbose) {
    FILE *file = fopen(filename, "r");
    int size;
    Data data;

    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    printf("Reading data: %s\n", filename);

    fscanf(file, "%d", &size);

    if (size < 0) {
        printf("Error: matrix size invalid\n");
        exit(1);
    }

    printf("Matrix size: %d\n", size);

    data.A = (double *)malloc(size * size * sizeof(double));
    data.b = (double *)malloc(size * sizeof(double));

    if (data.A == NULL || data.b == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(0);
    }

    for (int i = 0; i < size * size; i++) {
        fscanf(file, "%lf", &(data.A[i]));

        if (verbose < 1) continue;

        printf("%lf ", data.A[i]);
        if (i % size == size - 1) {
            printf("\n");
        }
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%lf", &(data.b[i]));
        if (verbose) printf("%lf ", data.b[i]);
    }
    if (verbose) printf("\n");

    fclose(file);

    data.size = size;

    return data;
}

Data init_data(int argc, char **argv, int n) {
    int size = n;
    char *filename = "sample.dat";
    int read_mode = 0;

    FILE *file;
    Data data;

    if (argc >= 2) {
        size = atoi(argv[1]);
    }
    if (argc >= 3) {
        filename = argv[2];
    }
    if (size == 0 || argc >= 3) {
        read_mode = 1;
    }

    if (read_mode) {
        file = fopen(filename, "r");

        if (file == NULL) {
            printf("Error opening file %s\n", filename);
            exit(1);
        }

        fscanf(file, "%d", &size);

        if (size < 0) {
            printf("Error: invalid matrix size\n");
            exit(1);
        }
    }

    data.A = (double *)malloc(size * size * sizeof(double));
    data.b = (double *)malloc(size * sizeof(double));
    data.x = (double *)malloc(size * sizeof(double));

    if (data.A == NULL || data.b == NULL || data.x == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(0);
    }

    if (read_mode) {
        printf("Reading data: %s\n", filename);

        for (int i = 0; i < size * size; i++) {
            fscanf(file, "%lf", &(data.A[i]));
        }

        for (int i = 0; i < size; i++) {
            fscanf(file, "%lf", &(data.b[i]));
        }

        fclose(file);
    } else {
        generate_diagonal_dominant_matrix(data.A, size, 0, 1200);
        generate_vector(data.b, size, 10, 200);
    }

    printf("Matrix size: %d\n", size);

    data.size = size;

    return data;
}

#endif  // GENERATE_MATRIX_H
