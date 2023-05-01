#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
typedef struct matrix{
  int rows, cols;
  double **data;
  int shallow;
} matrix;

typedef struct LUP{
  matrix *L;
  matrix *U;
  int *P;
  int n;
} LUP;

typedef struct{
  matrix X;
  matrix y;
} data;

data load_classification_data(char *images, char *label_file, int bias);
void free_data(data d);
data random_batch(data d, int n);
char *fgetl(FILE *fp);
matrix load_matrix(const char *fname);
void save_matrix(matrix m, const char *fname);

matrix make_identity_homography();
matrix make_translation_homography(float dx, float dy);

void free_matrix(matrix m);
double mag_matrix(matrix m);
matrix make_matrix(int rows, int cols);
matrix copy_matrix(matrix m);
double *sle_solve(matrix A, double *b);
matrix matrix_mult_matrix(matrix a, matrix b);
matrix matrix_elmult_matrix(matrix a, matrix b);
void print_matrix(matrix m);
double **n_principal_components(matrix m, int n);
void test_matrix();
matrix solve_system(matrix M, matrix b);
matrix matrix_invert(matrix m);
matrix random_matrix(int rows, int cols, double s);
matrix transpose_matrix(matrix m);
matrix axpy_matrix(double a, matrix x, matrix y);
#endif
