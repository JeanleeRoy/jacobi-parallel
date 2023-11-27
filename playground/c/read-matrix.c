#include <stdio.h>

#include "../../lib/utils.h"

int main(int argc, char **argv) {
    char *filename = "sample.dat";
    int size = 0;
    int is_test = 0;

    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n > 0) size = n;
    }
    if (argc >= 3) {
        filename = argv[2];
    }
    if (size == 0 || argc >= 3) {
        is_test = 1;
    }

    printf("Matrix program\n");

    double *A = NULL;
    double *b = NULL;

    Data data = read_data_from_file(filename, 0);

    A = data.A;
    b = data.b;
    size = data.size;

    if (A == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    // generate_diagonal_dominant_matrix(A, size, 0, 1200);
    // print_matrix_inline(A, size, size);

    free(A);
    free(b);

    return 0;
}
