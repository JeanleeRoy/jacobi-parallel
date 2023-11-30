#include <stdio.h>

#include "../../lib/utils.h"

// Compile: gcc -o generate-data.out generate-data.c -lm

#define N 100

int main(int argc, char const *argv[]) {
    int size = N;  // Size of the system of equations
    int is_fortran = 0;

    if (argc >= 2) {
        size = atoi(argv[1]);
    }
    if (argc == 3) {
        is_fortran = !!atoi(argv[2]);
    }

    double *A = (double *)malloc(size * size * sizeof(double));
    double *b = (double *)malloc(size * sizeof(double));

    generate_diagonal_dominant_matrix(A, size, 0, 1200);
    generate_vector(b, size, 10, 200);

    if (is_fortran) printf("\nC format:\n");
    printf("%d\n", size);
    print_array_inline(A, size * size, 0);
    print_vector_inline(b, size);

    if (is_fortran) {
        printf("\nFotran format:\n");
        printf("%d\n", size);
        print_array_inline(A, size * size, 1);
        print_vector_inline(b, size);
    }

    free(A);
    free(b);

    return 0;
}
