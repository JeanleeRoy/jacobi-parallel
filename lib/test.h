#ifndef TEST_H
#define TEST_H

void read_test_data(double *matrix, double *vec, int size, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%lf", &matrix[i]);
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%lf", &vec[i]);
    }

    fclose(file);
}

void read_test_matrix(double *matrix, int size, char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        fscanf(file, "%lf", &matrix[i]);
    }

    fclose(file);
}

#endif
