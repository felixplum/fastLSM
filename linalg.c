#include "linalg.h"
#include<stdio.h>



float min(float a, float b) {
    return a < b ? a : b;
};

float max(float a, float b) {
    return a >= b ? a : b;
};

floatMat* calloc_2D_fmat(size_t dimA, size_t dimB, char* name) {
    floatMat* ret = malloc(sizeof (floatMat));
    ret->name = name;
    ret->n_dim = 2;
    ret->shape = malloc(ret->n_dim*sizeof (size_t));
    ret->shape[0] = dimA;
    ret->shape[1] = dimB; 
    ret->data = (float *)calloc(dimA * dimB, sizeof(float));
    return ret;
}

floatMat* calloc_3D_fmat(size_t dimA, size_t dimB, size_t dimC, char* name) {
    floatMat* ret = malloc(sizeof (floatMat));
    ret->name = name;
    ret->n_dim = 3;
    ret->shape = malloc(ret->n_dim*sizeof (size_t));
    ret->shape[0] = dimA;
    ret->shape[1] = dimB;
    ret->shape[1] = dimC;
    ret->data = (float *)calloc(dimA * dimB * dimC, sizeof(float));
    return ret;
}

void print_2d_mat(floatMat* mat) {
    printf("Matrix: %s \n", mat->name);
    size_t n_rows =  mat->shape[0];
    size_t n_cols =  mat->shape[1];
    print_2d_(mat->data, n_rows, n_cols);
}

void mean(floatMat* mat) {
    
}

void print_2d_(float* arr, size_t n_rows, size_t n_cols) {
    float* row_ptr;
    for (size_t i = 0; i < n_rows; i++)
    {
        row_ptr = &(arr[i*n_cols]);
        for (size_t j = 0; j <  n_cols; j++)
        {
            printf("%2.f, ",row_ptr[j]);
        }
        printf("\n");
    }
}

void free_mat(floatMat* mat) {
   free(mat->data);
   free(mat->shape);
   free(mat); 
}


