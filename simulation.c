#include"simulations.h"

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

void init_dummy_data(float*strike_out, float* spots, size_t n_scens, size_t n_days)
{
    srand(1);
    float t;
    for (size_t day_i = 0; day_i < n_days; day_i++){
        t = 1.*(float)day_i / 365.;
        for (size_t i = 0; i < n_scens; i++)
        {
            spots[day_i*n_scens +i] = 20. + 1. * cos((float)day_i / 365 * 2 * M_PI) +t*(rand()%100)/100.;
            strike_out[day_i*n_scens+i] = 20.;
        }
    }
}
