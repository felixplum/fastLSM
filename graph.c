#include "graph.h"

#define NUM_ACTIONS_MAX 3

void init_state_containers(size_t num, stateContainer* containers,
      float* payments, float* costs, size_t n_scens) {
    for (size_t i = 0; i < num; i++)
    {
        containers[i].n_states = 0;
        containers[i].n_scens = n_scens;
        containers[i].payments = payments + i*n_scens;
        containers[i].costs = &(costs[i*n_scens]);
        containers[i].state_lb = NULL;
        containers[i].state_ub = NULL;
        if (i < num-1) containers[i].next = &(containers[i+1]);
        if (i > 0) containers[i].prev = &(containers[i-1]);
    }    
}

void free_state_containers(stateContainer* containers) {

}

void add_state_to_container(stateContainer* container, State* state, State* start_state_lookup) {
    // Todo: Check if node with this value exists already
    if (!container->state_lb || !container->state_ub) {
        container->state_lb = state;
        container->state_ub = state;
        container->n_states += 1;
        state->parent = container;
        return;
    }
    State* iter = container->state_ub;
    State* iter_prev = NULL;
    if(start_state_lookup) iter = start_state_lookup;
    // As long as there's a state:
    while(iter) {
        if (state->value == iter->value) {
            return;
        }
        if (state->value > iter->value) {
            state->state_down = iter;
            state->state_up = iter_prev;
            iter->state_up = state;
            if (iter->state_down) iter->state_down->state_up = state;
            container->n_states += 1;
            if (iter==container->state_ub) container->state_ub = state;
            break;
        }
        iter_prev = iter;
        iter = iter->state_down;
    }
    // we reached bottom without assigment; iter_prev holds last entry
    if (!iter && iter_prev) {
        iter_prev->state_down = state;
        state->state_up = iter_prev;
        container->state_lb = state;
        container->n_states += 1;
    }
    state->parent = container;
}

State* create_state(float value, float* actions, size_t num_scens) {
    State* ret_state = (State*)calloc(1, sizeof(State));
    ret_state->value = value;
    ret_state->actions = (float*)calloc(NUM_ACTIONS_MAX,sizeof(float));
    ret_state->transition_probs = (float*)calloc(NUM_ACTIONS_MAX,sizeof(float));
    ret_state->cv_coeff = (float*)calloc(4,sizeof(float));
    ret_state->ds_ds0 = 0.;
    ret_state->skip_node = false;
    ret_state->expected_value = 0.;
    ret_state->reachable_states = (State**)calloc(NUM_ACTIONS_MAX,sizeof(State*));
    ret_state->n_actions = sizeof(actions)/sizeof(actions[0]); // actually available actions
    memcpy(ret_state->actions, actions, sizeof(float) * ret_state->n_actions);
    ret_state->continuation_values = (float*)calloc(num_scens,sizeof(float));
    ret_state->state_up = NULL;
    ret_state->state_down = NULL;
    ret_state->parent = NULL;

    return ret_state;
}

void set_successor_state(State* from, State* to, size_t action_idx) {
   from->reachable_states[action_idx] = to;
}

void remove_state(State* state) {
    // update all connected components pointing to or from this state
    // free memory
}

void remove_action(State* state, float action) {
    // create tmp buffer with size prev-1
    // iterate over action, add if not equal to action
    for (size_t i = 0; i < state->n_actions; i++)
    {
        // search action
        if (state->actions[i] == action) {
            int n_actions_to_move = state->n_actions-i-1;
            if (n_actions_to_move > 0) {
                memmove(state->actions+i+1, state->actions+i, n_actions_to_move);
                memmove(state->transition_probs+i+1, state->transition_probs+i, n_actions_to_move);
            }
            state->n_actions--;
            return;
        }
    }
}