#include <stdio.h>
#include <time.h>
#include "linalg.h"
#include "graph.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

const int N_SCENS = 2000;
const int N_STEPS = 365;

typedef struct contractInfo {
    float dcq_min;
    float dcq_max;
    float tcq;
    float tcq_min_final;
} contractInfo;

double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}


// The cumulative normal distribution function
double CND( double X ){
    double L, K, w ;
    double const a1 = 0.31938153, a2 = -0.356563782, a3 = 1.781477937;
    double const a4 = -1.821255978, a5 = 1.330274429;
    L = fabs(X);
    K = 1.0 / (1.0 + 0.2316419 * L);
    w = 1.0 - 1.0 / sqrt(2 * M_PI) * exp(-L *L / 2) * (a1 * K + a2 * K *K + a3 * pow(K,3) + a4 * pow(K,4) + a5 * pow(K,5));
    if (X < 0 ){
        w= 1.0 - w;
    }
    return w;
}

double BlackScholes(char CallPutFlag, double S, double X, double T, double r, double v){
    // From: https://cseweb.ucsd.edu/~goguen/courses/130/SayBlackScholes.html
    // T in fraction of a year, X strike, S underlying, v annual std of returns
    double d1, d2;
    d1=(log(S/X)+(r+v*v/2)*T)/(v*sqrt(T));
    d2=d1-v*sqrt(T);
    if(CallPutFlag == 'c')
    return S *CND(d1)-X * exp(-r*T)*CND(d2);
    else if(CallPutFlag == 'p')
    return X * exp(-r * T) * CND(-d2) - S * CND(-d1);
}

double compute_option_strip(float r, float sigma, float strike, float s0, size_t n_days) {
    // Calls Black scholes, but un-discounts the value of the price of the option itself
    float sigma_annual = sigma;//*sqrt(365);
    float value = 0;
    float r_d = pow(1.+r, 1./365.); // daily risk-free return from annual one
    for (size_t i = 1; i <= n_days; i++)
    {
        float t_mat = (float)i/(float)n_days;
        value += pow(r_d, i)*BlackScholes('c', s0, strike, t_mat, r, sigma_annual);
        if (i==10) {
            float test = BlackScholes('c', s0, strike, t_mat, r, sigma_annual);
        }
    }
    return value;
}


void init_dummy_data_gbm(float*strike_out, float* spots, size_t n_scens, size_t n_days, float mu, float sigma)
{
    srand(1);
    float noise;
    float t = 0;
    const float dt = 1./(float)n_days;
    for (size_t day_i = 0; day_i < n_days; day_i++){
        t = day_i*dt;
        for (size_t i = 0; i < n_scens; i++)
        {
            spots[day_i*n_scens +i] = 20.*exp((mu-0.5*sigma*sigma)*t + sqrt(t)*sigma*randn(0, 1));
            if (day_i == 300) {
                printf("%.2f\n", spots[day_i*n_scens +i]);
            }
            strike_out[day_i*n_scens+i] = 20.;
        }
    }
}

// void init_dummy_data(float*strike_out, float* spots, size_t n_scens, size_t n_days)
// {
//     srand(1);
//     for (size_t day_i = 0; day_i < n_days; day_i++){
//         for (size_t i = 0; i < n_scens; i++)
//         {
//             spots[day_i*n_scens +i] = 20. + 1. * cos((float)day_i / 365 * 2 * M_PI) +0.1*(rand()%100)/100.;
//             strike_out[day_i*n_scens+i] = 20.;
//         }
//     }
// }


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
    for (size_t i = 0; i < n_scens; i++)
    {
        max_value = -1e10;
        strike_spot_diff = strikes[i] - spots[i];
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
    float params[4] = {0.,0.,0, 0.};
    if (parent_container->prev) {
        regress_cholesky(parent_container->prev->payments, state->continuation_values, params, n_scens, 1, 3, true);

    } else {
        regress_cholesky(spots, state->continuation_values, params, n_scens, 1, 3, true);
    }
        // regress_cholesky(spots, state->continuation_values, params, n_scens, 1, 3, true);


}

void optimize(stateContainer* containers, size_t n_scens) {
    size_t n_steps = N_STEPS;//sizeof(containers)/sizeof(containers);       TODO
    stateContainer* container_t;
    State* state_iter, *state_next;
    float expected_value;
    float* cont_values_interp = malloc(n_scens*sizeof(float)); // tmp data store
    float v_next;
    for (int t_i = n_steps-2; t_i >= 0; t_i--)
    {
        // if (t_i == 0) {
        //     float abc = 123;
        // }
        state_iter = containers[t_i].state_ub;
        // Iter over states top down
        while (state_iter)
        {
            update_contination_value(state_iter);
            // expected_value = mean_vec(state_iter->continuation_values, n_scens);
            // printf("t=%i, state=%i, expt: %.2f", t_i, state_idx, expected_value);
            state_iter = state_iter->state_down;
        }
    }
    printf("Value of contract %.2f\n", mean_vec(containers[0].state_ub->continuation_values, n_scens));
    free(cont_values_interp);
}

void test_regression()
{

    // float L[4] = {1,0,2,1};
    // float rhs[2] = {1, 5};
    // float out[2];
    // L_L_T_solve(L, rhs, out, 2);
    // print_vec(out, 2);
    // exit(0);
    const size_t n_s = 1000;
    float rf[n_s];
    float target[n_s];
    srand(4711);
    float x_sample, noise;
    for (size_t i = 0; i < n_s; i++)
    {
        // x_sample = (rand() % 30);
        // rf[i] = x_sample;
        // noise = (rand() % 100) / 100.;;
        // target[i] =  2000+ x_sample+10*x_sample*x_sample + 1000.*noise;

        x_sample = i;
        rf[i] = x_sample;
        target[i] =  x_sample*x_sample;
    }
    float params[3] = {0, 0, 0};
    // clock_t start, end;
    // double cpu_time_used;
    // start = clock();
    regress_cholesky(rf, target, params, n_s, 1, 2, false);
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
        .tcq_min_final = 0.
    };

    float* spot_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    float* strike_scens = (float*)malloc(N_SCENS*N_STEPS*sizeof(float));
    float mu = 0.3; float sigma = 0.0001;
    init_dummy_data_gbm(strike_scens, spot_scens, N_SCENS, N_STEPS, mu, sigma); 
    // printf("Opion value: %.2f\n", BlackScholes('c', 20., 20., 365./365., mu, sigma));
    float opt_strip_value = deal.dcq_max*compute_option_strip(mu, sigma, 20., 20., 365);
    printf("Opion value: %.2f\n",opt_strip_value);
    // exit(0);

    stateContainer* containers = calloc(N_STEPS, sizeof(stateContainer));
    init_states(N_STEPS, containers, spot_scens, strike_scens, N_SCENS, deal);

    optimize(containers, N_SCENS);

    // // /////////////////
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("It took %.5f ms", 1000 * cpu_time_used);
    free(spot_scens); free(strike_scens);
    return 0;
}
