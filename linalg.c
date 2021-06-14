#include "linalg.h"
#include<stdio.h>



float min(float a, float b) {
    return a < b ? a : b;
};

float max(float a, float b) {
    return a >= b ? a : b;
};

floatMat* calloc_2D_fmat(size_t dimA, size_t dimB, char* name) {
    floatMat* ret = malloc(sizeof (floatMat));
    ret->name = name;
    ret->n_dim = 2;
    ret->shape = malloc(ret->n_dim*sizeof (size_t));
    ret->shape[0] = dimA;
    ret->shape[1] = dimB; 
    ret->data = (float *)calloc(dimA * dimB, sizeof(float));
    return ret;
}

floatMat* calloc_3D_fmat(size_t dimA, size_t dimB, size_t dimC, char* name) {
    floatMat* ret = malloc(sizeof (floatMat));
    ret->name = name;
    ret->n_dim = 3;
    ret->shape = malloc(ret->n_dim*sizeof (size_t));
    ret->shape[0] = dimA;
    ret->shape[1] = dimB;
    ret->shape[1] = dimC;
    ret->data = (float *)calloc(dimA * dimB * dimC, sizeof(float));
    return ret;
}

void print_2d_mat(floatMat* mat) {
    printf("Matrix: %s \n", mat->name);
    size_t n_rows =  mat->shape[0];
    size_t n_cols =  mat->shape[1];
    print_2d_(mat->data, n_rows, n_cols);
}

void approx_mean_std(float* values, size_t n_samples, float* mean_out, float* std_out) {
    float sum = 0.;
    float sum_sq = 0.;
    for (size_t i = 0; i < n_samples; i++)
    {
        sum += values[i];
    }
    sum /= n_samples;
    for (size_t i = 0; i < n_samples; i++)
    {
        sum_sq += (sum-values[i])*(sum-values[i]);
    }
    
    *mean_out = sum;
    *std_out = sqrt(sum_sq/n_samples);
    
}

float *cholesky(float *A, int n) {
    float *L = (float*)calloc(n * n, sizeof(float));
    if (L == NULL)
        exit(EXIT_FAILURE);
 
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
            float s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (i == j) ?
                           sqrt(A[i * n + i] - s) :
                           (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
    return L;
}



float L_L_T_solve(float* L, float* rhs, float*out, size_t n) {
    // Solves LL^T*beta = rhs by fwd. and backward substituion
    float* b0 = malloc(n*sizeof(float));
    for (size_t i = 0; i < n; i++)
    {
        float res = rhs[i];
        for (size_t k = 0; k < i; k++)
        {
            res -= b0[k]*L[i*n+k];
        }
        b0[i] = res / L[i*n+i];
    }
    // Now solve L^T*beta = b0
    for (int i = n-1; i >= 0; i--)
    {
        float res = b0[i];
        for (int k = n-1; k > i; k--)
        {
            res -= out[k]*L[k*n+i];
        }
        out[i] = res / L[i*n+i];
    }
    free(b0);    
}


void regress_cholesky(float* risk_factors, float* target, float *params_out,
             size_t n_samples, size_t n_rf, size_t order, bool map_data) {

    static int cnt = 0;
    cnt+=1;
    bool debug = cnt == 30000;
    // float target_scale = 1.;// / 1000.;
    // float feature_scale = 1. / 1000.;
    float mean_out, std_out, mean_target, std_target;
    approx_mean_std(risk_factors, min(100, n_samples), &mean_out, &std_out);
    approx_mean_std(target, min(100, n_samples), &mean_target, &std_target);
    std_out += 1e-5; std_target += 1e-5;
    // Solve LLS beta = (X^TX)^1 * X^Ty by cholesky decomp.
    const size_t n_params = order+1;
    const size_t n_samples_reg = min(200, n_samples); 
    float* X = malloc(n_params*n_samples_reg*sizeof(float));
    float* X_T = malloc(n_params*n_samples_reg*sizeof(float));
    float* X_T_X = calloc(n_params*n_params,sizeof(float));
    float* rhs = calloc(n_samples_reg,sizeof(float));
    float* params_est = malloc(n_params*sizeof(float));
    // float* L = calloc(n_params*n_params,sizeof(float));
    // Compute feature vecs
    float x_;
    for (size_t i_row = 0; i_row < n_samples_reg; i_row++)
    {
        x_ = (risk_factors[i_row]-mean_out)/std_out;
        for (size_t i_col = 0; i_col < n_params; i_col++)
        {
            X[i_row*n_params+i_col] = pow(x_, i_col);
            X_T[i_col*n_samples_reg+i_row] = X[i_row*n_params+i_col];
        }
    }
    // printf("X: \n"); print_2d_(X, n_samples_reg, n_params);
    // printf("X^T: \n");print_2d_(X_T, n_params, n_samples_reg);

    // Compute rhs of LLS, i.e. X^Ty
    for (size_t i = 0; i < n_params; i++)
    {
        for (size_t k = 0; k < n_samples_reg; k++)
        {
            rhs[i] += X_T[i*n_samples_reg+k]*target[k] / std_target;
        }
    }

    // Compute X^T * X
    float tmp = 0.;
    float* row_ptr, *col_ptr;
    for (size_t i_row = 0; i_row < n_params; i_row++)
    {
        row_ptr = X_T + i_row*n_samples_reg;
        for (size_t i_col = 0; i_col < n_params; i_col++)
        {
            // Column of X is same as row of X_T
            col_ptr = X_T + i_col*n_samples_reg;
            tmp = 0.;
            for (size_t k = 0; k < n_samples_reg; k++)
            {
                tmp +=  row_ptr[k] * col_ptr[k];
            }
            if (i_row == i_col) tmp += 1e-6; // numerical stability

             X_T_X[i_row*n_params + i_col] = tmp;
        }
    } 

    // Compute cholesky decom. X_T_X = LL^T
    if (debug) {
        float teststop= 0;
    }
    float* L = cholesky(X_T_X, n_params);

    L_L_T_solve(L, rhs, params_out, n_params);
    if (debug) {
        // printf("X^TX: \n");print_2d_(X_T_X, n_params, n_params);
        // printf("L: \n"); print_2d_(L, n_params, n_params);
        // printf("RHS: \n"); print_vec(rhs, n_params);
        // printf("Params: \n");print_vec(params_out, n_params);
    }

    if (debug) {
            float error = 0.;
            float diff;
            int k = 0;
            float pred_i;
            float t_i;
            float x_;

            FILE * temp = fopen("data.temp", "w");

            for(int i=0; i < 1000; i++) {
                x_ = (risk_factors[i]-mean_out)/std_out;
                pred_i = (params_out[0] + params_out[1]*x_
                        + params_out[2]*x_*x_ + params_out[3]*x_*x_*x_)*std_target;
                t_i = (target[i]);
                diff = pred_i -  t_i;
                error += diff*diff;
                fprintf(temp, "%lf %lf %lf\n", risk_factors[i], target[i], pred_i);
                // fprintf(temp, "%lf %lf %lf\n", x_, t_i, pred_i*std_target+mean_target);

            }
            fclose(temp);  

            FILE *gnuplot = popen("gnuplot --persist", "w");
            // fprintf(gnuplot, "set style line 3 lt 1 lw 3 pt 3 lc rgb 'blue'\n");
            // fprintf(gnuplot, "set style line 2 lc rgb 'blue'\n");
            fprintf(gnuplot, "plot 'data.temp' using 1:3 lc rgb 'black'\n");
            fprintf(gnuplot, "replot 'data.temp' using 1:2 lc rgb 'red'\n");
            fflush(gnuplot);


            error = sqrt(error/n_samples);
            printf("MSE: %.3f\n", error);    
        }


   // Apply regression coeff. to original data
    if (map_data) {
        assert(n_rf==1);
        float x_;
        for(int i=0; i < n_samples; i++) {
            x_ = (risk_factors[i]-mean_out)/std_out;
            target[i] = (params_out[0] + params_out[1]*x_ + params_out[2]*x_*x_ + params_out[3]*x_*x_*x_)*std_target;

            // target[i] = params_out[0];// + params_out[1]*x_ + params_out[2]*x_*x_ + params_out[3]*x_*x_*x_;
            // for (size_t k = 1; k < n_params; k++)
            // {
            //     target[i] += params_out[k]*pow(x_, k);
            // }
        }
    }

    free(X);
    free(X_T);
    free(L);
    free(rhs);
}


void regress(float* risk_factors, float* target, float *params_out,
             size_t n_samples, size_t n_rf, size_t order, bool map_data) {
    // risk_factos is [n_sim x n_rf] matrix, where risk factors are not exponentiated
    // Computes the optimal parameters via stoch. gradient descent
    static int test = 0;
    test +=1;
    bool debug =false;// test == 40000;
    size_t n_params = n_rf*(order+1);
    float* rf_i;
    float target_i, rf_i_j;
    float* gradient_i = (float*)calloc(n_params, sizeof(float));
    float* regressors_i = (float*)calloc(n_params, sizeof(float));
    float* gradient_avg = (float*)calloc(n_params, sizeof(float));

    float* regressors_i_j, *params_j;
    float target_predict;
    float pred_error_i;
    // size_t batch_size = 1;
    float grad_sum_sq = 0.;
    float learning_rate =  0.2;
    float lr_decay = 0.99995;
    int n_iter = min(n_samples, 1000);
    // float lr_decay = pow(0.01, 1./((float)n_samples));
    
    // Numerical conditioning
    float mean_out, std_out, mean_target, std_target;
    approx_mean_std(risk_factors, min(100, n_samples), &mean_out, &std_out);
    approx_mean_std(target, min(100, n_samples), &mean_target, &std_target);
    std_target += 1e-5; std_out += 1e-5;
    // mean_target = 0;
    // std_target = 1.;
    // float target_scale =1.  / max_vec(target, n_samples);
    // float rf_scale = 1;//./ (max_vec(risk_factors, n_samples)-rf_offset);

    // if ((target_scale > 1.) || rf_scale > 1.) {
    //     target_scale = 1.;
    //     rf_scale = 1.;
    // }

    // Iterate over all samples
    for (size_t i = 0; i < n_iter; i++)
    {
        // Compute gradient for current sample
        target_i = (target[i]-mean_target) / std_target;
        rf_i = &(risk_factors[i*n_rf]); // points to rows containing rf at sample i
        target_predict = 0;
        // For each risk factor, build polynomial of order "order"
        for (size_t j = 0; j < n_rf; j++)
        {
            // target_i = p0_0 + p1_0*rf_i_0 + p2*rf_i_0² ... p0_1 + p1_1*rf_i_1 + p2*rf_i_1² +  =: f(p)
            // Cost J(p) = (target_i-f(p))²
            // dJ(p)/dp = -2*(target_i-f(p))*(df(p)/dp)
            // With df(p)/dp =  [1, rf_i_0, rf_i_0², 1, rf_i_1, rf_i_1², ...]

            // Pointers to current rf's polynomial for convencience
            rf_i_j = (rf_i[j]-mean_out) / std_out;
            regressors_i_j = &(regressors_i[j*(order+1)]);
            params_j = &(params_out[j*(order+1)]);
            // gradient_i_j = &(gradient_i[j*(order+1)]); 

            // Build regressor (= gradient) + eval function
            regressors_i_j[0] = 1.; // const. term, i.e. p0_i
            target_predict += params_j[0];
            for (size_t order_k = 1; order_k < order+1; order_k++)
            {
                regressors_i_j[order_k] = regressors_i_j[order_k-1]*rf_i_j;
                target_predict += params_j[order_k]*regressors_i_j[order_k];
            }
        }
        if(debug) {
            printf("Target pred: %2.f; target: %.2f, cost: %.2f\n", target_predict, target_i, (target_predict-target_i)*(target_predict-target_i));
            printf("Regressors: "); print_vec(regressors_i_j, 3);
            printf("Params: "); print_vec(params_j, 3);
        }


        // Now compute final gradient, i.e. -2*(target_i-f(p))*(df(p)/dp)
        pred_error_i = 2*(target_predict-target_i);
        grad_sum_sq = 1e-8;
        // Compute magnitude of current gradient
        for (size_t param_idx = 0; param_idx < n_params; param_idx++)
        {
            // gradient_i_j[param_idx] = regressors_i[param_idx]*pred_error_i;
            gradient_i[param_idx] = regressors_i[param_idx]*pred_error_i;

            gradient_avg[param_idx] = 0.8*gradient_avg[param_idx] + 0.2*gradient_i[param_idx]*gradient_i[param_idx];
            grad_sum_sq += gradient_avg[param_idx]*gradient_avg[param_idx];
        }
        // Second pass: Update global gradient with normalized gradient
        grad_sum_sq = sqrt(grad_sum_sq);
        // printf("Grad tmp: "); print_vec(gradient_i_j, 3);
        // Do parameter step with accumulated gradient
        // if ((i % batch_size) == 0) {
        learning_rate *= lr_decay;
        // norm_vec(gradient_i, n_params);
        if(debug) {
            printf("Batch grad: "); print_vec(gradient_i, 3);
            printf("Params_j before update: "); print_vec(params_j, 3);
        }

        // printf("Batch grad: "); print_vec(gradient_i, 3);
        // printf("Params_j before update: "); print_vec(params_j, 3);
        for (size_t param_idx = 0; param_idx < n_params; param_idx++)
        {
            params_out[param_idx] -=  learning_rate*(gradient_i[param_idx]/grad_sum_sq);
            gradient_i[param_idx] = 0.; // reset
        }
        if(debug) {
            printf("Params_j after update: "); print_vec(params_j, 3);
            printf("\n_________________\n");
        }
        // printf("Params_j after update: "); print_vec(params_j, 3);
        // printf("LR %.3f\n", learning_rate);
        float a = 1;
            // printf("Grad: "); print_vec(gradient_i, 3);
        // }

        // printf("\n_________________\n");
    }
    // Do final iteration; access average error
    
    // Rescale parameters
    // scale_vec(params_out, n_params, 1./target_scale);
    
    // Plot
    if (debug) {
        float error = 0.;
        float diff;
        int k = 0;
        float pred_i;
        float t_i;
        float x_;

        FILE * temp = fopen("data.temp", "w");

        for(int i=0; i < 1000; i++) {
            x_ = (risk_factors[i]-mean_out) / std_out;
            pred_i = params_out[0] + params_out[1]*x_
                    + params_out[2]*x_*x_  + params_out[3]*x_*x_*x_;
            t_i = (target[i]-mean_target) / std_target;
            diff = pred_i -  t_i;
            error += diff*diff;
            fprintf(temp, "%lf %lf %lf\n", risk_factors[i], target[i], pred_i*std_target+mean_target);
            // fprintf(temp, "%lf %lf %lf\n", x_, t_i, pred_i*std_target+mean_target);

        }
        fclose(temp);  

        FILE *gnuplot = popen("gnuplot", "w");
        // fprintf(gnuplot, "set style line 3 lt 1 lw 3 pt 3 lc rgb 'blue'\n");
        // fprintf(gnuplot, "set style line 2 lc rgb 'blue'\n");
        fprintf(gnuplot, "plot 'data.temp' using 1:3 lc rgb 'black'\n");
        fprintf(gnuplot, "replot 'data.temp' using 1:2 lc rgb 'red'\n");
        fflush(gnuplot);


        error = sqrt(error/n_samples);
        printf("MSE: %.3f\n", error);    
    }

    // Apply regression coeff. to original data
    if (map_data) {
        assert(n_rf==1);
        float x_;
        for(int i=0; i < n_samples; i++) {
            // TODO: Do in generic way
            x_ = (risk_factors[i]-mean_out) / std_out;
            target[i] = params_out[0] + params_out[1]*x_ + params_out[2]*x_*x_ + params_out[3]*x_*x_*x_;
            // for (size_t k = 1; k < n_params; k++)
            // {
            //     target[i] += params_out[k]*pow(x_, k);
            // }
            target[i] = target[i]*std_target+mean_target;
        }
    }

    // cleanup
    free(gradient_i);
    free(regressors_i_j);
    free(gradient_avg);

}

void print_2d_(float* arr, size_t n_rows, size_t n_cols) {
    float* row_ptr;
    for (size_t i = 0; i < n_rows; i++)
    {
        row_ptr = &(arr[i*n_cols]);
        for (size_t j = 0; j <  n_cols; j++)
        {
            printf("%.2f, ",row_ptr[j]);
        }
        printf("\n");
    }
}

float max_vec(float* vec, size_t length) {
    float max = -1e20;
    for (size_t i = 0; i < length; i++)
    {
        if (vec[i] > max) max = vec[i];
    }
    return max;
}
float min_vec(float* vec, size_t length) {
    float min = 1e20;
    for (size_t i = 0; i < length; i++)
    {
        if (vec[i] < min) min = vec[i];
    }
    return min;
}

float mean_vec(float* vec, size_t length) {
    float sum=0;
    for (size_t i = 0; i < length; i++)
    {
        sum += vec[i];
    }
    return sum / (float)length;
}

void print_vec(float* vec, size_t length) {
    for (size_t i = 0; i < length; i++)
    {
        printf("%.3f, ",vec[i]);
    }
    printf("\n");
}

void norm_vec(float* vec, size_t length) {
    float sum_sq = 0.;
    for (size_t i = 0; i < length; i++)
    {
        /* code */
        sum_sq += vec[i]*vec[i];
    }
    sum_sq = sqrt(sum_sq);
    for (size_t i = 0; i < length; i++)
    {
        /* code */
        vec[i] = (vec[i]+1e-10)/(sum_sq+1e-10);
    }    
}

void scale_vec(float* vec, size_t length, float scale) {
    for (size_t i = 0; i < length; i++)
    {
        vec[i] *= scale;
    }
}
void free_mat(floatMat* mat) {
   free(mat->data);
   free(mat->shape);
   free(mat); 
}

