#include <stdio.h>
#include <time.h>
#include "linalg.h"
#include "graph.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "simulations.h"

const int N_SCENS = 10000;
const int N_STEPS = 365;

typedef struct contractInfo {
    float dcq_min;
    float dcq_max;
    float tcq;
    float tcq_min_final;
} contractInfo;



void init_states(size_t num_days, stateContainer* containers,
                 float* spot_scens, float* strike_scens, size_t num_scens,
                 contractInfo contract_info) {

    init_state_containers(num_days, containers, spot_scens, strike_scens, num_scens);

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
                value_next = tmp_state_from->value + actions[i_action];
                // If action allowed: Create state at t+1
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
    }
    printf("states: %i\n", n_states_total);

}

void mark_nodes_to_skip(stateContainer* containers, size_t num_days, bool consider_probabilities) {
    // Set first node in first container
    float value_next;
    size_t n_states_total = 0;
    State* tmp_state_from;
    float reduce_fraction = 0.8;
    float threshold;
    srand(1);
    size_t n_skipped = 0;
    for (size_t t_i = 0; t_i < num_days - 1; t_i++)
    {
        // For all states, apply actions to generate new states
        // Start with upper-bound state and go downwards
        tmp_state_from = containers[t_i].state_ub;
        while (tmp_state_from)
        {
            // Only consider inside nodes for removal
            if ((tmp_state_from != containers[t_i].state_ub) && tmp_state_from != containers[t_i].state_lb) {
                // in $reduce_fraction percent of the cases, mark node for skipping
                threshold = (rand() % 100 ) / 100.;
                if (threshold < reduce_fraction) {
                    tmp_state_from->skip_node = true;
                    n_skipped += 1;
                }
            }
            tmp_state_from = tmp_state_from->state_down;
        }
    }
    printf("Skipped %i nodes\n", n_skipped);
}


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
    float* cont_state = state->continuation_values;
    size_t best_action_idx;
    float expected_value = 0.;
    float* coeff;
    float delta = 0.;
    State* next_valid_node;
    for (size_t i = 0; i < n_scens; i++)
    {
        max_value = -1e10;
        strike_spot_diff = strikes[i] - spots[i];
        for (size_t action_i = 0; action_i < state->n_actions; action_i++)
        {
            next_valid_node = get_next_computed_node(state->reachable_states[action_i]);
            cont_i = (next_valid_node->continuation_values)[i];
            // cont_i = (state->reachable_states[action_i]->continuation_values)[i];
            v_next = state->value + state->actions[action_i];
            payoff0 = cont_i + strike_spot_diff*(state->actions[action_i]);
            if (payoff0 > max_value) {
                max_value = payoff0;
                best_action_idx = action_i;
            }
        }
        state->transition_probs[best_action_idx] += 1./(float)n_scens;
        // Compute delta
        //coeff = state->reachable_states[best_action_idx]->cv_coeff;
        //delta += -state->actions[best_action_idx];//  + coeff[1] + 2*coeff[2]*spots[i] + 3*coeff[3]*spots[i]*spots[i];
        cont_state[i] = max_value;
        expected_value += max_value;
        // update continuation for current volume and time in-place
    }
    // printf("%.3f %.3f\n",state->transition_probs[0], state->transition_probs[1]);
    state->expected_value = expected_value / (float)n_scens;
    float params[4] = {0.,0.,0, 0.};
    if (parent_container->prev) {
        regress_cholesky(parent_container->prev->payments, state->continuation_values, params, n_scens, 1, 3, true);

    } else {
        regress_cholesky(spots, state->continuation_values, params, n_scens, 1, 3, true);
    }

    state->delta = delta / (float)n_scens;

    memcpy(state->cv_coeff, params, 4);
        // regress_cholesky(spots, state->continuation_values, params, n_scens, 1, 3, true);
}

void writeMat(float* mat, size_t rows, size_t cols) {

   FILE *fp;
   size_t n = rows*cols;
   fp = fopen("mat_out.txt", "w+");
   for (size_t i = 0; i < n; i++)
   {
        fprintf(fp, "%.5f\n", mat[i]);
   }
   fclose(fp);
}

void optimize(stateContainer* containers, size_t n_scens) {
    size_t n_steps = N_STEPS;//sizeof(containers)/sizeof(containers);       TODO
    stateContainer* container_t;
    State* state_iter, *state_next;
    float expected_value;
    float v_next;

    // Backward pass
    for (int t_i = n_steps-2; t_i >= 0; t_i--)
    {
        state_iter = containers[t_i].state_ub;
        // Iter over states top down
        while (state_iter)
        {
            if (!state_iter->skip_node) {
                update_contination_value(state_iter);
            }
            // expected_value = mean_vec(state_iter->continuation_values, n_scens);
            // printf("t=%i, state=%i, expt: %.2f", t_i, state_idx, expected_value);
            state_iter = state_iter->state_down;
        }
    }
    printf("Value of contract %.2f\n", mean_vec(containers[0].state_ub->continuation_values, n_scens));
    // Compute sensitivites
    // float* prob_matrix = calloc(n_steps*365, sizeof(float));
    // containers[0].state_ub->ds_ds0 = 1.;
    // float delta_total = 0.;
    // for (size_t t_i = 0; t_i < n_steps-1; t_i++)
    // {
    //     state_iter = containers[t_i].state_ub;
    //     // Iter over states top down
    //     size_t node_idx = 0;
    //     float delta_avg = 0.;
    //     while (state_iter)
    //     {
    //         for (size_t action_i = 0; action_i < state_iter->n_actions; action_i++)
    //         {
    //             state_iter->reachable_states[action_i]->ds_ds0 += 
    //                 state_iter->ds_ds0 * state_iter->transition_probs[action_i];
    //         }
    //         // float mean, std;
    //         // approx_mean_std(state_iter->continuation_values, n_scens, &mean, &std);
    //         // (mean*3*std)*
    //         state_iter->delta *= state_iter->ds_ds0;
    //         delta_avg += state_iter->delta;
    //         prob_matrix[t_i*365+node_idx] = state_iter->ds_ds0;
    //         node_idx += 1;
    //         // Next node
    //         state_iter = state_iter->state_down;
    //     } 
    //     delta_avg = delta_avg;
    //     delta_total += delta_avg;
    //     // printf("Delta t_i = %i is: %.3f\n", t_i, delta_avg);
    // }
    // writeMat(prob_matrix, 365, n_steps);
    // printf("Total delta is: %.3f\n", delta_total);

    // free(prob_matrix);   
}

//
int main()
{

    clock_t start, end;
    double cpu_time_used;
    // test_regression();
    // exit(0);

    contractInfo deal = {
        .dcq_min = 0.,
        .dcq_max = 100,
        .tcq = 365*100,
        .tcq_min_final = 0.
    };

    float* spot_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    float* strike_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    
    // init_dummy_data(strike_scens, spot_scens, N_SCENS, N_STEPS); 
    
    float mu = 0.2; float sigma = 0.2;
    init_dummy_data_gbm(strike_scens, spot_scens, N_SCENS, N_STEPS, mu, sigma); 
    // printf("Opion value: %.2f\n", BlackScholes('c', 20., 20., 365./365., mu, sigma));
    float opt_strip_value = deal.dcq_max*compute_option_strip(mu, sigma, 20., 20., 365);
    float delta = deal.dcq_max*(compute_option_strip(mu, sigma, 20., 20.01, 365) - compute_option_strip(mu, sigma, 20., 20., 365))/0.01;
    printf("Option value: %.2f\n",opt_strip_value);
    printf("Option delta is %.3f\n", delta);
    // exit(0);

    stateContainer* containers = calloc(N_STEPS, sizeof(stateContainer));
    init_states(N_STEPS, containers, spot_scens, strike_scens, N_SCENS, deal);
    start = clock();
    
    mark_nodes_to_skip(containers, N_STEPS, false);
    optimize(containers, N_SCENS);
    end = clock();

    // // /////////////////
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms", 1000 * cpu_time_used);
    free(spot_scens); free(strike_scens);
    return 0;
}


// void test_regression()
// {

//     // float L[4] = {1,0,2,1};
//     // float rhs[2] = {1, 5};
//     // float out[2];
//     // L_L_T_solve(L, rhs, out, 2);
//     // print_vec(out, 2);
//     // exit(0);
//     const size_t n_s = 1000;
//     float rf[n_s];
//     float target[n_s];
//     srand(4711);
//     float x_sample, noise;
//     for (size_t i = 0; i < n_s; i++)
//     {
//         // x_sample = (rand() % 30);
//         // rf[i] = x_sample;
//         // noise = (rand() % 100) / 100.;;
//         // target[i] =  2000+ x_sample+10*x_sample*x_sample + 1000.*noise;

//         x_sample = i;
//         rf[i] = x_sample;
//         target[i] =  x_sample*x_sample;
//     }
//     float params[3] = {0, 0, 0};
//     // clock_t start, end;
//     // double cpu_time_used;
//     // start = clock();
//     regress_cholesky(rf, target, params, n_s, 1, 2, false);
//     // end = clock();
//     // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//     // printf("It took %.5f ms\n", 1000*cpu_time_used);
//     // for (size_t i = 0; i < 3; i++)
//     // {
//     //     printf("Param %i: %.2f \n", i, params[i]);
//     // }
// }