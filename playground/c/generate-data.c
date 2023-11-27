#include <stdio.h>

#include "../../lib/utils.h"

#define N 100

int main(int argc, char const *argv[]) {
    int size = N;  // Size of the system of equations

    if (argc == 2) {
        size = atoi(argv[1]);
    }

    double *A = (double *)malloc(size * size * sizeof(double));
    double *b = (double *)malloc(size * sizeof(double));
    
    generate_diagonal_dominant_matrix(A, size, 0, 1200);
    generate_vector(b, size, 10, 200);

    printf("%d\n", size);
    print_array_inline(A, size * size);
    print_array_inline(b, size);
    
    return 0;
}
