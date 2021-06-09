#include <stdio.h>
#include <time.h>
#include "linalg.h"
#include "graph.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

const int N_SCENS = 10000;
const int N_STEPS = 365;

typedef struct contractInfo {
    float dcq_min;
    float dcq_max;
    float tcq;
    float tcq_min_final;
} contractInfo;



void init_dummy_data(float*strike_out, float* spots, size_t n_scens, size_t n_days)
{
    srand(1);
    for (size_t day_i = 0; day_i < n_days; day_i++){
        for (size_t i = 0; i < n_scens; i++)
        {
            spots[day_i*n_scens +i] = 20. + 1. * cos((float)day_i / 365 * 2 * M_PI) +0.1*(rand()%100)/100.;
            // if (day_i > 182)
            //     strike_out[day_i*n_scens+i] = 0;//20.;
            // else strike_out[day_i*n_scens+i] = 1;
            strike_out[day_i*n_scens+i] = 19;
        }
    }
}

void init_states(size_t num_days, stateContainer* containers,
                 float* spot_scens, float* strike_scens, size_t num_scens,
                 contractInfo contract_info) {
    // Malloc and basic init:
    // clock_t start, end;
    // double cpu_time_used;
    // start = clock();

    init_state_containers(num_days, containers, spot_scens, strike_scens, num_scens);
    // end = clock();
    // cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("It took %.5f ms for init state containers\n", 1000 * cpu_time_used);
    float actions[2] = {-contract_info.dcq_min, -contract_info.dcq_max};
    size_t num_actions = sizeof(actions)/sizeof(actions[0]);
    // Set first node in first container
    State* init_state = create_state(contract_info.tcq, actions, num_scens);
    State* tmp_state_from, *tmp_state_to, *state_last;
    float value_next;
    size_t n_states_total = 0;
    add_state_to_container(&(containers[0]), init_state, NULL);
    for (size_t t_i = 0; t_i < num_days - 1; t_i++)
    {
        // For all states, apply actions to generate new states
        // Start with upper-bound state and go downwards
        tmp_state_from = containers[t_i].state_ub;
        state_last = NULL;
        while (tmp_state_from)
        {
            for (size_t i_action = 0; i_action < num_actions; i_action++)
            {
                // apply action i to state
                value_next = tmp_state_from->value + actions[i_action]; //, contract_info.tcq_min_final);
                if (value_next >= contract_info.tcq_min_final) {
                    if (!state_last || (state_last && state_last->value != value_next)) {
                        tmp_state_to = create_state(value_next, actions, num_scens);
                        add_state_to_container(&(containers[t_i+1]), tmp_state_to, state_last);
                        set_successor_state(tmp_state_from, tmp_state_to, i_action);
                        state_last = tmp_state_to;
                    } else if (state_last && state_last->value == value_next)
                    {
                        // state exists, but still needs to be set as successor
                        set_successor_state(tmp_state_from, state_last, i_action);
                    }
                    // printf("add to states t=%i: %.2f\n", t_i+1, value_next);
                // action not allowed, remove from action set
                } else {
                    // TODO: Only mark for removal and do in the end; otherwise bug when removing multiple actions
                    remove_action(tmp_state_from, actions[i_action]);
                }
            }
            tmp_state_from = tmp_state_from->state_down;
        }
        n_states_total += containers[t_i].n_states;
        // printf("states: %i\n", containers[t_i].n_states);
        //  &containers[t_i];
        // Iter over number of actions to create nodes in next state
    }
    printf("states: %i\n", n_states_total);

}

// void get_interpolated_values(State* node_from, float lookup_value, float* result, size_t n_scens) {

// }

void update_contination_value(State* state) {
    // Our action will lead us here
    stateContainer* parent_container = state->parent;
    float* spots = parent_container->payments;
    float* strikes = parent_container->costs;
    size_t n_scens = parent_container->n_scens;
    float v_next;
    float payoff0, payoff1, strike_spot_diff, cont_i, max_value;
    float* actions = state->actions;
    // We might have to interpolate, though, TODO
    // float** cont_value_lut;

    // float* cont_vals_action_0 = (state->reachable_states[0]->continuation_values);
    // float* cont_vals_action_1 = (state->reachable_states[1]->continuation_values);
    float* cont_state = state->continuation_values;
    for (size_t i = 0; i < n_scens; i++)
    {
        // max_value = -1e10;
        strike_spot_diff = strikes[i] - spots[i];
        // payoff0 = cont_vals_action_0[i] + strike_spot_diff*actions[0];
        // payoff1 = cont_vals_action_1[i] + strike_spot_diff*actions[1];
        // if (payoff0 > payoff1) 
        //     cont_state[i] = payoff0; 
        // else cont_state[i] = payoff1;
        for (size_t action_i = 0; action_i < state->n_actions; action_i++)
        {
            cont_i = (state->reachable_states[action_i]->continuation_values)[i];
            v_next = state->value + state->actions[action_i];
            payoff0 = cont_i + strike_spot_diff*(state->actions[action_i]);
            if (payoff0> max_value) max_value = payoff0;
        }
        cont_state[i] = max_value;
            // update continuation for current volume and time in-place
    }
    float params[3] = {0.,0.,0};
    regress(spots, state->continuation_values, params, n_scens, 1, 2, true);
}

void optimize(stateContainer* containers, size_t n_scens) {
    size_t n_steps = N_STEPS;//sizeof(containers)/sizeof(containers);       TODO
    stateContainer* container_t;
    State* state_iter, *state_next;
    float* cont_values_interp = malloc(n_scens*sizeof(float)); // tmp data store
    float v_next;
    for (int t_i = n_steps-2; t_i >= 0; t_i--)
    {
        state_iter = containers[t_i].state_ub;
        // Iter over states top down
        while (state_iter)
        {
            update_contination_value(state_iter);
            // print_vec(state_iter->continuation_values, n_scens);
            if(state_iter == state_iter->state_down) {
                float a = 12.;
            }
            state_iter = state_iter->state_down;
        }
    }
    printf("Value of contract %.2f\n", containers[0].state_ub->continuation_values[0]);
    free(cont_values_interp);
}

void test_regression()
{
    const size_t n_s = 1000;
    float rf[n_s];
    float target[n_s];
    srand(4711);
    float x_sample, noise;
    for (size_t i = 0; i < n_s; i++)
    {
        x_sample = (rand() % 30);
        rf[i] = x_sample;
        noise = (rand() % 100) / 100.;;
        target[i] =  2000+ x_sample+10*x_sample*x_sample + 1000.*noise;
    }
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

//
int main()
{

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    // test_regression();
    // exit(0);

    contractInfo deal = {
        .dcq_min = 0.,
        .dcq_max = 100,
        .tcq = 365*100,
        .tcq_min_final = 200*100.
    };

    float* spot_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    float* strike_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    init_dummy_data(strike_scens, spot_scens, N_SCENS, N_STEPS);
    stateContainer* containers = malloc(N_STEPS * sizeof(stateContainer));
    init_states(N_STEPS, containers, spot_scens, strike_scens, N_SCENS, deal);

    //optimize(containers, N_SCENS);
    /////////////////
    // floatMat *continuation_value = calloc_3D_fmat(N_STEPS, N_GRID, N_SCENS, "Continuation value");
    // floatMat *volumes = calloc_2D_fmat(N_STEPS, N_GRID, "Volumes");
    // floatMat *strike_out = calloc_2D_fmat(N_STEPS, N_SCENS, "Strikes");
    // floatMat *spots = calloc_2D_fmat(N_STEPS, N_SCENS, "Spots");

    // print_2d_mat(retMat);
    // init_dummy_data(strike_out, spots);
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
