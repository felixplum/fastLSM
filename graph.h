
#ifndef GRAPH_   /* Include guard */
#define GRAPH_

#include<stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h> 
#include <assert.h>
#include <string.h>

typedef struct State State;
typedef struct stateContainer stateContainer;

struct State {
        // Information for a given state
    float value;                // Value of the state (i.e. after discretization of state space)
    float* continuation_values; // Continuation value per scenario
    float* actions;             // admissable set
    float* transition_probs;    // Probability for actions[i]
    float* cv_coeff;            // Fitting params, such that CV(state_k) = Ax
    float cv_expected;          // Expected value of the continuation fn
    size_t n_actions;           // Number of actions
    State* reachable_states;    // States[i] reachable by taking actions[i]
    State* state_up;                // neighbour in state container with higher value
    State* state_down;                // neighbour in state container with lower value
};

struct stateContainer {
    size_t n_states;        // Number of states
    State* state_ub;        // Upper bound state
    State* state_lb;        // Lower bound state
    stateContainer* next;
    stateContainer* prev;
    float* costs;           // Costs per scenario (i.e. Strike); there might be costs-per-action-type later on
    float* payments;        // Paypement per scenarion (i.e. Spot price)
};

void init_state_containers(size_t num, stateContainer* containers);
void free_state_containers(stateContainer* containers);
void remove_state(State* state);
void add_state_to_container(stateContainer* container, State* state, State* start_state_lookup);
State* create_state(float value, float* actions, size_t num_scens);

#endif 