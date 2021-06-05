#include <stdio.h>
#include <time.h>
#include "linalg.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

const int N_SCENS = 10000;
const int N_GRID = 16;
const int N_STEPS = 365;

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

// float continuation_value[num_days][N_GRID][N_SCENS];
// float volumes[num_days][N_GRID];
// float strike_out[num_days][N_SCENS];
// float spots[num_days][N_SCENS];

void init_volume_grid(floatMat *volumes)
{
    // Example swing contract: 0% ToP
    float DCQ_MIN = 0;
    float DCQ_MAX = 100;
    float TCQ = 365 * DCQ_MAX;
    float TCQ_MIN_FINAL = 0;
    float TCQ_MAX_FINAL = TCQ;
    float min_prev = TCQ;
    float max_prev = TCQ;
    float min_curr, max_curr;
    float incr;
    size_t n_steps = volumes->shape[0];
    size_t n_grid = volumes->shape[1];
    for (size_t t_i = 0; t_i < n_steps; t_i++)
    {
        // Only valid for Swing, i.e. withdrawal only
        min_curr = max(TCQ_MIN_FINAL, min_prev - DCQ_MAX);
        max_curr = min(TCQ_MAX_FINAL, max_prev - DCQ_MIN);
        min_prev = min_curr;
        max_prev = max_curr;
        incr = (max_curr - min_curr) / (n_grid - 1);
        float *v_start_arr = &(volumes->data[t_i * n_grid]); // TODO: Check if written correctly
        for (int v_idx = 0; v_idx < n_grid; v_idx++)
        {
            v_start_arr[v_idx] = min_curr + v_idx * incr;
            // if (t_i == 0) printf("Time %i: %.2f\n", t_i, v_start_arr[v_idx]);
        }
    }
}

void init_dummy_data(floatMat *strike_out, floatMat *spots)
{
    size_t n_rows = spots->shape[0];
    size_t n_cols = spots->shape[1];
    //assert(n_cols == 1);
    for (size_t i = 0; i < n_rows; i++)
    {
        spots->data[i * n_cols + 0] = 20. + 1. * cos((float)i / 365 * 2 * M_PI);
        strike_out->data[i * n_cols + 0] = 20.;
    }

    // print_vec(spots->data, 365);
    // print_vec(strike_out->data, 365);

}

// void compute_volume_interp_params(float* v0_in, float* v1_in, int* idx_offset_out, float* alpha_out) {
//     // such that v0[i+offset] = alpha * v1[i + offset] + (1-alpha)*v1(i + offset+1)
//     // Assumes equidistant grid
//     // Called once per time-step
//     // Can then be used to compute interpolated continuation value between two vectors;
//     // first vector represents vector after a const. decision has been applied to all elements (e.g. one of DdCQ_MIN, DCQ_MAX)
// }

void test_regression()
{
    const size_t n_s = 1000;
    float rf[n_s];
    float target[n_s];
    // srand(4711);
    float x_sample, noise;
    // for (size_t i = 0; i < n_s; i++)
    // {
    //     x_sample = (rand() % 30);
    //     rf[i] = x_sample;
    //     noise = (rand() % 100) / 100.;;
    //     target[i] =  4.5*x_sample-70*x_sample*x_sample + 100.*noise;
    // }
    float params[3] = {0, 0, 0};
    // clock_t start, end;
    // double cpu_time_used;
    // start = clock();
    for (size_t i = 0; i < 1; i++)
    {
        regress(rf, target, &params, n_s, 1, 2, false);
    }
    // end = clock();
    // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // printf("It took %.5f ms\n", 1000*cpu_time_used);
    // for (size_t i = 0; i < 3; i++)
    // {
    //     printf("Param %i: %.2f \n", i, params[i]);
    // }
}

// void test_malloc() {
//     clock_t start, end;
//     double cpu_time_used;
//     start = clock();
//     size_t n = 50;
//     float** res_ptr = malloc(n*sizeof(float*));
//     for (size_t i = 0; i < n; i++)
//     {
//         res_ptr[i] = calloc(2*2*512*10000,sizeof(float));
//         for (size_t k = 0; k < 2*2*512*10000; k++)
//         {
//             res_ptr[i][k] = k; 
//         }
        
//     }
//     end = clock();
//     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     printf("It took %.5f ms to allocate\n", 1000*cpu_time_used);
//     start = clock();
//     for (size_t i = 0; i < n; i++)
//     {
//         free(res_ptr[i]);
//     }
//     end = clock();
//     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     printf("It took %.5f ms to free\n", 1000*cpu_time_used);
    
//     free(res_ptr);
// }

int interp(float *lookup_grid, size_t n_entries, float lookup_val, float *alpha_out, float *shift_out)
{
    if (lookup_val > lookup_grid[n_entries - 1] || lookup_val < lookup_grid[0])
    {
        *alpha_out = 0.;
        *shift_out = -1;
        return 1; // continuation value is 0, since volume out of bounds / not permitted
    }
    else if (lookup_val == lookup_grid[n_entries - 1])
    {
        *alpha_out = 1.;
        *shift_out = n_entries - 2;
        return 0;
    }
    size_t i;
    for (i = 0; i < n_entries; ++i)
    {
        if (lookup_val >= lookup_grid[i] && lookup_val < lookup_grid[i+1])
        {
            break;
        }
    }
    *alpha_out = (lookup_val - lookup_grid[i]) / (lookup_grid[i + 1] - lookup_grid[i]);
    *shift_out = i;
    return 0;
    //target_grid[i] + alpha * (target_grid[i+1]  - target_grid[i] );
}

void compute_immediate_payoff(float *spots_t, float *strikes_t, const float *decisions_t, float *result,
                              size_t n_scens, size_t n_decisions)
{
    // Stores dv*(S_i-K_i) in matrix of dim [n_dec x n_scens]
    float decision;
    for (size_t i_dec = 0; i_dec < n_decisions; i_dec++)
    {
        decision = decisions_t[i_dec];
        for (size_t i_scen = 0; i_scen < n_scens; i_scen++)
        {
            // withdrawal of decision of -100 ITM leads to positive payoff -300*(strike-spot) = 300 * (spot - strike)
            result[i_dec * n_scens + i_scen] = decision * (strikes_t[i_scen] - spots_t[i_scen]);
        }
        // print_vec(&(result[i_dec*n_scens]), n_scens);
    }
}

void compute_volume_interp_lookup(float *volumes_t, float *volumes_t_next, float *decisions, float *result, size_t n_grid, size_t n_dec)
{
    // Computes scale and offset for all [volume state X decision] from grid(t) -> grid(t+1)
    // These coefficients can then be used to interpolate between the respective continuation values
    // result[0] := alpha; result[1] = offset
    float v_lookup;
    float alpha_interp;
    int offset_interp;
    for (size_t v_i = 0; v_i < n_grid; v_i++)
    {
        for (size_t dec_i = 0; dec_i < n_dec; dec_i++)
        {
            v_lookup = volumes_t[v_i] + decisions[dec_i];
            interp(volumes_t_next, n_grid, v_lookup, &(result[v_i * n_dec * 2 + 2*dec_i]), &(result[v_i * n_dec * 2 + 2*dec_i + 1]));
        }
    }
}

void optimize(floatMat *continuation_value, floatMat *volumes,
              floatMat *strike_out, floatMat *spots)
{
    // continuation_value: [steps * grid_size * scens]
    // volumes: [steps * grid_size] Hold the allowed volumes, discretized
    // strike, spot: [steps * scens]

    size_t n_days = spots->shape[0];
    size_t n_scens = spots->shape[1];
    size_t n_grid = volumes->shape[1];

    // These point to the matrices at time t
    float *cont_t, *cont_t_next, *volumes_t, *volumes_t_next, *strikes_t,
            *spots_t, *immediate_returns_t, *expected_cont_value_t;
    // tmp vars:
    float payoff_t_i;

    // Last entries set; now do backward iteration
    //////////////////////////////////////////////////
    float v_t, v_t_next;
    int n_decisions = 2;
    float decisions[2] = {0, -300}; // ONLY FOR TESTING
    floatMat *expected_values = calloc_2D_fmat(n_days, n_grid, "Expected continuation value per (t,v_k)");
    floatMat *immediate_returns = calloc_3D_fmat(n_days, n_scens, n_decisions, "Immediate returns");
    floatMat *interps_cache = calloc_3D_fmat(n_grid, n_decisions, 2, "tmp");
    for (int t_i = n_days - 2; t_i >= 0; t_i--)
    {
        // Starting from volume level vector v_i [n_grid] at time t_i, compute
        // the two volume vectors v_dcq0_i+1, v_dcq1_i+1 that result from either
        // taking DCQ_0 or DCQ_1. Per vector, we can compute a set of interpolation params
        // Telling us how to interpolate between two sets of continuation value scenarios, in order
        // To obtain a set of continuation value scenarios per volume level

        // To each set, we add the immediate payoff resulting from DCQ_0 or DCQ1
        // Per decision, we obtain an immediate payoff-per-scenario
        strikes_t = &(strike_out->data[t_i * n_scens]);
        spots_t = &(spots->data[t_i * n_scens]);
        volumes_t = &(volumes->data[t_i * n_grid]);
        expected_cont_value_t = &(expected_values->data[t_i * n_grid]);
        volumes_t_next = &(volumes->data[(t_i + 1) * n_grid]);
        cont_t = &(continuation_value->data[(t_i)*n_scens * n_grid]);
        cont_t_next = &(continuation_value->data[(t_i + 1) * n_scens * n_grid]);
        immediate_returns_t = &(immediate_returns->data[(t_i)*n_scens * n_decisions]);
        // print_vec(volumes_t, n_grid);
        // print_vec(volumes_t_next, n_grid);

        compute_immediate_payoff(spots_t, strikes_t, decisions, immediate_returns_t, n_scens, n_decisions);
        float alpha_interp;
        int offset_interp;
        float cont_val_i;
        float max_value, max_dec;
        compute_volume_interp_lookup(volumes_t, volumes_t_next, decisions, interps_cache->data, n_grid, n_decisions);
        // Iterate over state space, i.e. volume grid at time t
        // int interp_idx;
        for (size_t v_i = 0; v_i < n_grid; v_i++)
        {
            v_t = volumes_t[v_i];
            test_regression();
            for (size_t scen_i = 0; scen_i < n_scens; scen_i++)
            {
                // expected_value[1] = 0.;
                max_value = -1e10;
                for (size_t dec_i = 0; dec_i < 2; dec_i++)
                {
                    alpha_interp = interps_cache->data[v_i * n_decisions * 2 + dec_i*2];
                    offset_interp = interps_cache->data[v_i * n_decisions * 2 + dec_i*2 + 1];
                    if (offset_interp > 0 ) {
                        v_t_next = v_t + decisions[dec_i];
                        cont_val_i = cont_t_next[offset_interp * n_scens + scen_i] +
                                    alpha_interp * (cont_t_next[(offset_interp+1) * n_scens + scen_i] - cont_t_next[offset_interp * n_scens + scen_i]);
                        payoff_t_i = immediate_returns_t[dec_i * n_scens + scen_i];
                        // printf("%.2f %.2f %.2f\n",v_t_next,cont_val_i, payoff_t_i);
                        if ((cont_val_i + payoff_t_i) > max_value)
                        {
                            max_value = cont_val_i + payoff_t_i;
                            max_dec = decisions[dec_i];
                        }
                    } else {
                        // Lookup failed, i.e. outside of grid
                    }

                }
                // update continuation for current volume and time in-place
                cont_t[v_i * n_scens + scen_i] = max_value;
                expected_cont_value_t[v_i] += max_value;
            }
            expected_cont_value_t[v_i] /= (float)(n_scens);
        }
        // print_vec(expected_cont_value_t, n_grid);
    }
    // print_2d_mat(expected_values);
    free_mat(immediate_returns);
    free_mat(interps_cache);
    free_mat(expected_values);
}

//
int main()
{

    // test_regression();
    test_malloc();
    exit(0);
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    /////////////////
    floatMat *continuation_value = calloc_3D_fmat(N_STEPS, N_GRID, N_SCENS, "Continuation value");
    floatMat *volumes = calloc_2D_fmat(N_STEPS, N_GRID, "Volumes");
    floatMat *strike_out = calloc_2D_fmat(N_STEPS, N_SCENS, "Strikes");
    floatMat *spots = calloc_2D_fmat(N_STEPS, N_SCENS, "Spots");

    // print_2d_mat(retMat);
    init_dummy_data(strike_out, spots);
    init_volume_grid(volumes);
    // print_2d_mat(volumes);

    optimize(continuation_value, volumes, strike_out, spots);

    // /////////////////
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms", 1000 * cpu_time_used);

    free_mat(continuation_value);
    free_mat(volumes);
    free_mat(strike_out);
    free_mat(spots);
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