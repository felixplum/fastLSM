#ifndef LINALG_   /* Include guard */
#define LINALG_

#include<stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 
#include <assert.h>

typedef struct floatMat {
    size_t* shape;
    size_t n_dim;
    float* data;
    char* name;
} floatMat;

void free_mat(floatMat* mat);

void regress(float* risk_factors, float* target, float *params_out,
             size_t n_samples, size_t n_rf, size_t order, bool map_data);
void approx_mean_std(float* values, size_t n_samples, float* mean_out, float* std_out);
float min(float a, float b);
float max(float a, float b);
float max_vec(float* vec, size_t length);
float min_vec(float* vec, size_t length);
void scale_vec(float* vec, size_t length, float scale);
void print_vec(float* vec, size_t length);
void norm_vec(float* vec, size_t length);
void print_2d_mat(floatMat* mat);
void print_2d_(float* arr, size_t n_rows, size_t n_cols);
floatMat* calloc_2D_fmat(size_t dimA, size_t dimB, char* name);
floatMat* calloc_3D_fmat(size_t dimA, size_t dimB, size_t dimC, char* name);
#endif 