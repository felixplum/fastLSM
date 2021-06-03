#ifndef LINALG_   /* Include guard */
#define LINALG_

#include<stdio.h>
#include <stdlib.h>

typedef struct floatMat {
    size_t* shape;
    size_t n_dim;
    float* data;
    char* name;
} floatMat;

void free_mat(floatMat* mat);
float min(float a, float b);
float max(float a, float b);
void print_2d_mat(floatMat* mat);
void print_2d_(float* arr, size_t n_rows, size_t n_cols);
floatMat* calloc_2D_fmat(size_t dimA, size_t dimB, char* name);
floatMat* calloc_3D_fmat(size_t dimA, size_t dimB, size_t dimC, char* name);
#endif 