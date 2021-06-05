#include <stdio.h>
#include <time.h>
#include "linalg.h"
#include "graph.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

const int N_SCENS = 10000;
const int N_GRID = 16;
const int N_STEPS = 365;

typedef struct contractInfo {
    float dcq_min;
    float dcq_max;
    float tcq;
    float tcq_min_final;
} contractInfo;

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

void init_dummy_data(float*strike_out, float* spots, size_t n_scens, size_t n_days)
{

    for (size_t i = 0; i < n_scens; i++)
    {
        spots[i * n_days + 0] = 20. + 1. * cos((float)i / 365 * 2 * M_PI);
        strike_out[i * n_days + 0] = 20.;
    }
}

void init_states(size_t num_days, stateContainer* containers,
                 float* spot_scens, float* strike_scens, size_t num_scens,
                 contractInfo contract_info) {
    // Malloc and basic init:
    // clock_t start, end;
    // double cpu_time_used;
    // start = clock();

    init_state_containers(num_days, containers);
    // end = clock();
    // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("It took %.5f ms for init state containers\n", 1000 * cpu_time_used);
    float actions[2] = {-contract_info.dcq_min, -contract_info.dcq_max};
    // Set first node in first container
    State* init_state = create_state(contract_info.tcq, actions, num_scens);
    State* tmp_state_from, *tmp_state_to, *state_last;
    float value_next;
    add_state_to_container(&(containers[0]), init_state, NULL);
    for (size_t t_i = 0; t_i < num_days - 1; t_i++)
    {
        // For all states, apply actions to generate new states
        tmp_state_from = containers[t_i].state_ub;
        state_last = NULL;
        while (tmp_state_from)
        {
            
            for (size_t i_action = 0; i_action < sizeof(actions)/sizeof(actions[0]); i_action++)
            {
                // apply action i to state
                value_next = max(tmp_state_from->value + actions[i_action], contract_info.tcq_min_final);
                if (!state_last || (state_last && state_last->value != value_next)) {
                    tmp_state_to = create_state(value_next, actions, num_scens);
                    add_state_to_container(&(containers[t_i+1]), tmp_state_to, state_last);
                    state_last = tmp_state_to;
                }
                // printf("add to states t=%i: %.2f\n", t_i+1, value_next);
            }
            tmp_state_from = tmp_state_from->state_down;
        }
        // printf("states: %i\n", containers[t_i].n_states);
        //  &containers[t_i];
        // Iter over number of actions to create nodes in next state
    }
}



//
int main()
{

    // test_malloc();
    // test_regression();
    clock_t start, end;
    double cpu_time_used;
    start = clock();


    contractInfo deal = {
        .dcq_min = 0.,
        .dcq_max = 100,
        .tcq = 365*100,
        .tcq_min_final = 0.
    };

    float* spot_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    float* strike_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    init_dummy_data(strike_scens, spot_scens, N_SCENS, N_STEPS);
    stateContainer* containers = malloc(N_STEPS * sizeof(stateContainer));
    init_states(N_STEPS, containers, spot_scens, strike_scens, N_SCENS, deal); 
    /////////////////
    // floatMat *continuation_value = calloc_3D_fmat(N_STEPS, N_GRID, N_SCENS, "Continuation value");
    // floatMat *volumes = calloc_2D_fmat(N_STEPS, N_GRID, "Volumes");
    // floatMat *strike_out = calloc_2D_fmat(N_STEPS, N_SCENS, "Strikes");
    // floatMat *spots = calloc_2D_fmat(N_STEPS, N_SCENS, "Spots");

    // print_2d_mat(retMat);
    // init_dummy_data(strike_out, spots);
    // init_volume_grid(volumes);
    // // print_2d_mat(volumes);

    // optimize(continuation_value, volumes, strike_out, spots);

    // // /////////////////
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms", 1000 * cpu_time_used);

    // free_mat(continuation_value);
    // free_mat(volumes);
    // free_mat(strike_out);
    // free_mat(spots);
    free(spot_scens); free(strike_scens);
    return 0;
}