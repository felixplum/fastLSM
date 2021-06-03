#include<stdio.h>
#include <time.h>
#include "linalg.h"
#include <math.h> 
#include <stdlib.h>
#include <assert.h>

const int N_SCENS = 8;
const int N_GRID = 16;
const int N_STEPS = 365;

#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))



// float continuation_value[num_days][N_GRID][N_SCENS];
// float volumes[num_days][N_GRID];
// float strike_out[num_days][N_SCENS];
// float spots[num_days][N_SCENS];

void init_volume_grid(floatMat* volumes) {
    // Example swing contract: 0% ToP
    float DCQ_MIN = 0;
    float DCQ_MAX = 100;
    float TCQ = 365*DCQ_MAX;
    float TCQ_MIN_FINAL = 0;
    float TCQ_MAX_FINAL = TCQ;
    float min_prev = TCQ;
    float max_prev = TCQ;
    float min_curr, max_curr;
    float incr;
    size_t n_steps = volumes->shape[0];
    size_t n_grid = volumes->shape[1];
    for (size_t t_i = 0; t_i < n_steps; t_i++) {
        // Only valid for Swing, i.e. withdrawal only
        min_curr = max(TCQ_MIN_FINAL, min_prev - DCQ_MAX);
        max_curr = min(TCQ_MAX_FINAL, max_prev - DCQ_MIN);
        min_prev = min_curr;
        max_prev = max_curr;
        incr = (max_curr-min_curr) / (n_grid - 1);
        float* v_start_arr = &(volumes->data[t_i*n_grid]);                     // TODO: Check if written correctly
        for (int v_idx = 0; v_idx < n_grid; v_idx++) {
            v_start_arr[v_idx] = min_curr + v_idx*incr;
            // if (t_i == 0) printf("Time %i: %.2f\n", t_i, v_start_arr[v_idx]);
        }
    }
}

void init_dummy_data(floatMat* strike_out, floatMat* spots) {
    size_t n_rows = spots->shape[0];
    size_t n_cols = spots->shape[1];
    //assert(n_cols == 1);
    for (size_t i = 0; i < n_rows; i++)
    {
        spots->data[i*n_cols + 0] = 20. + 5*sin((float)i/365*2*M_PI);
        strike_out->data[i*n_cols + 0]= 20.;
    }
}

// void compute_volume_interp_params(float* v0_in, float* v1_in, int* idx_offset_out, float* alpha_out) {
//     // such that v0[i+offset] = alpha * v1[i + offset] + (1-alpha)*v1(i + offset+1)
//     // Assumes equidistant grid
//     // Called once per time-step
//     // Can then be used to compute interpolated continuation value between two vectors;
//     // first vector represents vector after a const. decision has been applied to all elements (e.g. one of DdCQ_MIN, DCQ_MAX)
// }

void test_regression() {
    float rf[100000];
    float target[100000];
    for (size_t i = 0; i < 100000; i++)
    {
        rf[i] = i;
        target[i] = i*i;
    }
    
    float params[3] = {0,1.,0};
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    regress(rf, target, &params, 100000, 1, 2); 
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms\n", 1000*cpu_time_used);
    for (size_t i = 0; i < 3; i++)
    {
        printf("Param %i: %.2f \n", i, params[i]);
    }
    
}

void print_vec(float* vec, size_t length) {
    for (size_t i = 0; i < length; i++)
    {
        printf("%.3f, ",vec[i]);
    }
    printf("\n");
}

void regress(float* risk_factors, float* target, float *params_out, size_t n_samples, size_t n_rf, size_t order) {
    // risk_factos is [n_sim x n_rf] matrix, where risk factors are not exponentiated
    // Computes the optimal parameters via stoch. gradient descent
    
    float* rf_i;
    float target_i, rf_i_j;
    float* gradient_i = (float*)calloc(n_rf*(order+1), sizeof(float));
    float* gradient_i_j = (float*)calloc(n_rf*(order+1), sizeof(float));
    float* regressors_i = (float*)calloc(n_rf*(order+1), sizeof(float));
    float* regressors_i_j, *params_j;
    float target_predict;
    float pred_error_i;
    size_t batch_size = 1;
    float grad_sum_sq = 0.;
    float learning_rate = 1.;
    
    // Iterate over all samples
    for (size_t i = 0; i < n_samples; i++)
    {
        // Compute gradient for current sample
        target_i = target[i];
        rf_i = &(risk_factors[i*n_rf]); // points to rows containing rf at sample i
        target_predict = 0;
        // For each risk factor, build polynomial of order "order"
        for (size_t j = 0; j < n_rf; j++)
        {
            // target_i = p0_0 + p1_0*rf_i_0 + p2*rf_i_0² ... p0_1 + p1_1*rf_i_1 + p2*rf_i_1² +  =: f(p)
            // Cost J(p) = (target_i-f(p))²
            // dJ(p)/dp = -2*(target_i-f(p))*(df(p)/dp)
            // With df(p)/dp =  [1, rf_i_0, rf_i_0², 1, rf_i_1, rf_i_1², ...]

            // Pointers to current rf's polynomial for convencience
            rf_i_j = rf_i[j];
            regressors_i_j = &(regressors_i[j*(order+1)]);
            params_j = &(params_out[j*(order+1)]);
            // gradient_i_j = &(gradient_i[j*(order+1)]); 

            // Build regressor (= gradient) + eval function
            regressors_i_j[0] = 1.; // const. term, i.e. p0_i
            target_predict += params_j[0];
            for (size_t order_k = 1; order_k < order+1; order_k++)
            {
                regressors_i_j[order_k] = regressors_i_j[order_k-1]*rf_i_j;
                target_predict += params_j[order_k]*regressors_i_j[order_k];
            }
        }
        // printf("Target pred: %2.f; target: %.2f, cost: %.2f", target_predict, target_i, (target_predict-target_i)*(target_predict-target_i));
        // printf("Regresors: "); print_vec(regressors_i_j, 3);
        // printf("Params: "); print_vec(params_j, 3);

        // Now compute final gradient, i.e. -2*(target_i-f(p))*(df(p)/dp)
        pred_error_i = 2*(target_predict-target_i);
        grad_sum_sq = 0.;
        // Compute magnitude of current gradient
        for (size_t param_idx = 0; param_idx < n_rf*(order+1); param_idx++)
        {
            gradient_i_j[param_idx] = regressors_i[param_idx]*pred_error_i;
            grad_sum_sq += gradient_i_j[param_idx]*gradient_i_j[param_idx];
        }
        // Second pass: Update global gradient with normalized gradient
        grad_sum_sq = sqrt(grad_sum_sq);
        for (size_t param_idx = 0; param_idx < n_rf*(order+1); param_idx++)
        {
            gradient_i[param_idx] +=  gradient_i_j[param_idx] / (grad_sum_sq+1e-8);
        }
        // printf("Grad: "); print_vec(gradient_i, 3);
        // Do parameter step with accumulated gradient
        if ((i % batch_size) == 0) {
            learning_rate = exp(-(float)i*0.01);
            for (size_t param_idx = 0; param_idx < n_rf*(order+1); param_idx++)
                {
                    params_out[param_idx] -=  learning_rate*gradient_i[param_idx]/((float)batch_size);
                    gradient_i[param_idx] = 0.; // reset
                }
            // printf("Params_j after update: "); print_vec(params_j, 3);
            // printf("Grad: "); print_vec(gradient_i, 3);
            
        }
    }
    
    // cleanup
    free(gradient_i);
    free(gradient_i_j);
    free(regressors_i_j);

}

int interp(float* lookup_grid, size_t n_entries, float lookup_val, float* alpha_out, int* shift_out) {
    if (lookup_val > lookup_grid[n_entries-1] || lookup_val < lookup_grid[0] ) {
        return 1; // continuation value is 0, since volume out of bounds / not permitted
    } else if (lookup_val == lookup_grid[n_entries-1] ) {
        *alpha_out = 1.;
        *shift_out = -1;
        return 0;
    }
    size_t i;
    for (i = 0; i < n_entries; i++)
    {
        if (lookup_val >= lookup_grid[i]) {
            break;
        }
    }
    *alpha_out = (lookup_val - lookup_grid[i]) / (lookup_grid[i+1] - lookup_grid[i]);
    *shift_out = i;
    return 0;//target_grid[i] + alpha * (target_grid[i+1]  - target_grid[i] );
}

void compute_immediate_returns(float* spots_t, float* strikes_t, const float* decisions_t, float* result,
                               size_t n_scens, size_t n_decisions) {
    float decision;
    for (size_t i_dec = 0; i_dec < n_decisions; i_dec++)
    {
        decision = decisions_t[i_dec];
        for (size_t i_scen = 0; i_scen < n_scens; i_scen++)
        {
            result[i_dec*n_scens + i_scen] = decision*(spots_t[i_scen] - strikes_t[i_scen]);
        } 
    }
}

void compute_volume_interp_lookup(float* volumes_t, float* volumes_t_next, float* decisions, float* result, size_t n_grid, size_t n_dec) {
    float v_lookup;
    float alpha_interp;
    int offset_interp;
    for (size_t v_i = 0; v_i < n_grid; v_i++)
    {
        for (size_t dec_i = 0; dec_i < n_dec; dec_i++)
        {
            v_lookup = volumes_t[v_i] + decisions[dec_i];
            result[dec_i] = interp(volumes_t_next, n_grid, v_lookup, &alpha_interp, &offset_interp);            // TODO
            // TODO
        }
        
    }
    
}

void optimize(floatMat* continuation_value, floatMat* volumes,
              floatMat* strike_out, floatMat*spots) {
    // continuation_value: [steps * grid_size * scens] 
    // volumes: [steps * grid_size] Hold the allowed volumes, discretized 
    // strike, spot: [steps * scens] 

    size_t n_days = spots->shape[0];
    size_t n_scens = spots->shape[1]; 
    size_t n_grid = volumes->shape[1];

    // These point to the matrices at time t
    float *cont_t, *cont_t_next, *volumes_t, *volumes_t_next, *strikes_t, *spots_t, *immediate_returns_t;
    // tmp vars:
    float payoff_t_i;

    // Last entries set; now do backward iteration
    //////////////////////////////////////////////////
    float v_t, v_t_next;
    const int n_decisions = 2;
    const float decisions[2] = {0, -300};                                                                                 // ONLY FOR TESTING
    float expected_value[2];
    floatMat* cont_t_tmp[2];
    cont_t_tmp[0] = calloc_2D_fmat(continuation_value->shape[0], continuation_value->shape[1], "cont tmp0");
    cont_t_tmp[1] = calloc_2D_fmat(continuation_value->shape[0], continuation_value->shape[1], "cont tmp1");
    floatMat* immediate_returns = calloc_3D_fmat(n_days, n_scens, n_decisions, "Immediate returns");
    floatMat* interps_tmp = calloc_3D_fmat(n_grid, n_decisions, 2, "tmp");
    for (int t_i = n_days-2; t_i >= 0; t_i--) {
        // Starting from volume level vector v_i [n_grid] at time t_i, compute
        // the two volume vectors v_dcq0_i+1, v_dcq1_i+1 that result from either 
        // taking DCQ_0 or DCQ_1. Per vector, we can compute a set of interpolation params
        // Telling us how to interpolate between two sets of continuation value scenarios, in order 
        // To obtain a set of continuation value scenarios per volume level

        // To each set, we add the immediate payoff resulting from DCQ_0 or DCQ1
        // Per decision, we obtain an immediate payoff-per-scenario
        strikes_t = &(strike_out->data[t_i*n_scens]);
        spots_t = &(spots->data[t_i*n_scens]);
        volumes_t = &(volumes->data[t_i*n_grid]);
        volumes_t_next = &(volumes->data[(t_i+1)*n_grid]);
        cont_t = &(continuation_value->data[(t_i)*n_scens*n_grid]);
        cont_t_next = &(continuation_value->data[(t_i+1)*n_scens*n_grid]);
        immediate_returns_t = &(immediate_returns->data[(t_i)*n_scens*n_decisions]);
        compute_immediate_returns(spots_t, strikes_t, decisions, immediate_returns_t, n_scens, n_decisions);
        float alpha_interp;
        int offset_interp;
        float cont_val_tmp;
        compute_volume_interp_lookup(volumes_t, volumes_t_next, decisions, interps_tmp->data, n_grid, n_decisions); // TODO: pass correct result
        // Iterate over state space, i.e. volume grid at time t
        for (size_t v_i = 0; v_i < n_grid; v_i++) {
            v_t = volumes_t[v_i];


            for (size_t scen_i = 0;  scen_i < n_scens; scen_i++) {
                // expected_value[0] = 0.;
                // expected_value[1] = 0.;
                float max_value = -1e10;
                float max_dec = 0.;
                for (size_t dec_i = 0; dec_i < 2; dec_i++) {
                    v_t_next = v_t + decisions[dec_i];
                    //if (interp(volumes_t_next, n_grid, v_t_next, &alpha_interp, &offset_interp) == 0) { // TODO: Put interpolation in outer loop
                        cont_val_tmp = cont_t_next[v_i*n_scens + scen_i + offset_interp] + 
                        alpha_interp*(cont_t_next[v_i*n_scens + scen_i + offset_interp+1]-cont_t_next[v_i*n_scens + scen_i + offset_interp]);
                        payoff_t_i = immediate_returns_t[dec_i*n_scens+dec_i];
                        cont_t_tmp[dec_i]->data[v_i*n_scens + scen_i] = cont_val_tmp + payoff_t_i;
                        if ((cont_val_tmp + payoff_t_i) >  max_value) {
                            max_value = cont_val_tmp + payoff_t_i;
                            max_dec = decisions[dec_i];
                        }
                    //}
                }
                
            }

            if (expected_value[0] > expected_value[1]) {
                // first action led to better expected value, apply & update continuation value
            }
            //printf("%.2f, %.2f", expected_value[0], expected_value[1]);
            

        }


    }
    free_mat(immediate_returns);
    free_mat(interps_tmp);
}

// 
int main() {

    test_regression();
    exit(0);
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    /////////////////
    floatMat* continuation_value = calloc_3D_fmat(N_STEPS, N_GRID, N_SCENS, "Continuation value");
    floatMat* volumes = calloc_2D_fmat(N_STEPS, N_GRID, "Volumes");
    floatMat* strike_out = calloc_2D_fmat(N_STEPS, N_SCENS, "Strikes");
    floatMat* spots = calloc_2D_fmat(N_STEPS, N_SCENS, "Spots");


    // print_2d_mat(retMat);
    init_dummy_data(strike_out, spots);
    init_volume_grid(volumes);
    // print_2d_mat(volumes);

    optimize(continuation_value, volumes, strike_out, spots);

    // /////////////////
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms", 1000*cpu_time_used);

    // free_mat(continuation_value);
    // free_mat(volumes);
    // free_mat(strike_out);
    // free_mat(spots);
    return 0;
}



    // Initialize continuation value at last day
    ////////////////////////////////////////////////////
    // strikes_t = &(strike_out->data[(n_days-1)*n_scens]);
    // spots_t = &(spots->data[(n_days-1)*n_scens]);
    // cont_t = &(continuation_value->data[(n_days-1)*n_scens*n_grid]);
    // for (size_t scen_i = 0; scen_i < n_scens; scen_i++)
    // {
    //     // Determine payoff; max(0, spot-strike)
    //     if (spots_t[scen_i] > strikes_t[scen_i]) {
    //         payoff_t_i = spots_t[scen_i] - strikes_t[scen_i];
    //     } else {
    //         payoff_t_i = 0.;
    //     }
    //     // For all volume levels, we'll have the same continuation value
    //     for (size_t v_i = 0; v_i < n_grid; v_i++)
    //     {
    //         cont_t[v_i*n_scens + scen_i] = payoff_t_i;
    //     }
    // }
    // print_2d_(cont_t, n_grid, n_scens);