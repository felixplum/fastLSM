
#ifndef SIMULATIONS_   /* Include guard */
#define SIMULATIONS_

#include<stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 
#include <assert.h>
#include <string.h>

double randn (double mu, double sigma);

// The cumulative normal distribution function
double CND( double X );

double BlackScholes(char CallPutFlag, double S, double X, double T, double r, double v);

double compute_option_strip(float r, float sigma, float strike, float s0, size_t n_days);

void init_dummy_data_gbm(float*strike_out, float* spots, size_t n_scens, size_t n_days, float mu, float sigma);

void init_dummy_data(float*strike_out, float* spots, size_t n_scens, size_t n_days);

#endif