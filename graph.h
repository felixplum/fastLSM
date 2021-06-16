
#ifndef GRAPH_   /* Include guard */
#define GRAPH_

#include<stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 
#include <assert.h>
#include <string.h>
#include "linalg.h"

typedef struct State State;
typedef struct stateContainer stateContainer;

struct State {
        // Information for a given state
    float value;                // Value of the state (i.e. after discretization of state space)
    float* continuation_values; // Continuation value per scenario
    float* actions;             // admissable set
    float* transition_probs;    // Probability for actions[i]
    float* cv_coeff;            // Fitting params, such that CV(state_k) = Ax
    float  delta;
    float expected_value;          // Expected value of the continuation fn
    float ds_ds0;                // State probability = Sensitivity dS_i/dS0 of this node
    size_t n_actions;           // Number of actions
    State** reachable_states;    // States[i] reachable by taking actions[i]
    State* state_up;                // neighbour in state container with higher value
    State* state_down;                // neighbour in state container with lower value
    stateContainer* parent;     
};

struct stateContainer {
    size_t n_states;        // Number of states
    size_t n_scens;
    State* state_ub;        // Upper bound state
    State* state_lb;        // Lower bound state
    stateContainer* next;
    stateContainer* prev;
    float* costs;           // Costs per scenario (i.e. Strike); there might be costs-per-action-type later on
    float* payments;        // Paypement per scenarion (i.e. Spot price)
    float payments_mean;
};

void init_state_containers(size_t num, stateContainer* containers,
                           float* payments, float* costs, size_t n_scens);
void free_state_containers(stateContainer* containers);
void remove_state(State* state);
void set_successor_state(State* from, State* to, size_t action_idx);
void add_state_to_container(stateContainer* container, State* state, State* start_state_lookup);
void remove_action(State* state, float action);
State* create_state(float value, float* actions, size_t num_scens);

#endif 