#include "graph.h"

#define NUM_ACTIONS_MAX 10

void init_state_containers(size_t num, stateContainer* containers) {
    for (size_t i = 0; i < num; i++)
    {
        containers[i].n_states = 0;
        containers[i].state_lb = NULL;
        containers[i].state_ub = NULL;
        if (i < num-1) containers[i].next = &(containers[i+1]);
        if (i > 0) containers[i+1].prev = &(containers[i]);
    }    
}

void free_state_containers(stateContainer* containers) {

}

void add_state_to_container(stateContainer* container, State* state, State* start_state_lookup) {
    // Todo: Check if node with this value exists already
    if (!container->state_lb) {
        container->state_lb = state;
        container->state_ub = state;
        container->n_states += 1;
        return;
    }
    State* iter = container->state_ub;
    State* iter_prev;
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
    if (!iter) {
        iter_prev->state_down = state;
        state->state_up = iter_prev;
        container->state_lb = state;
        container->n_states += 1;
    }
}

State* create_state(float value, float* actions, size_t num_scens) {
    State* ret_state = (State*)malloc(sizeof(State));
    ret_state->value = value;
    ret_state->actions = (float*)malloc(NUM_ACTIONS_MAX*sizeof(float));
    ret_state->n_actions = sizeof(actions)/sizeof(actions[0]);
    memcpy(ret_state->actions, actions, ret_state->n_actions);
    ret_state->continuation_values = (float*)malloc(num_scens*sizeof(float));
    return ret_state;
}

void remove_state(State* state) {
    // update all connected components pointing to or from this state
    // free memory
}