data
{

   int I;               # Number of systems
   int J;               # Number of experts in a group
   int K;               # Number of groups

   vector[I] m[J,K];    # Experts' point estimates (calibration data)
   vector[I] e[J,K];    # Experts' uncertainty estimates (calibration data)
   vector[I] x;         # Known parameter values (calibration data)

   real mhat[J,K];      # Experts' point estimates for the parameter in a new system
   real ehat[J,K];      # Experts' uncertainty estimate for the parameter in a new system

   real s2p[4];
   real linvp[4];
   real rhop[4];
   real s2_barp[4];
   real linv_barp[4];
   real mup[4];

}

parameters
{

   # unknown parameter value in a new system
   real<lower=0,upper=1> xhat;

   # parameters
   real<lower=rhop[3],upper=rhop[4]> rho[J,K];
   vector<lower=mup[3],upper=mup[4]>[I+1] mu[J,K];
   real<lower=linvp[3],upper=linvp[4]> linv[J,K];
   real<lower=s2p[3],upper=s2p[4]> s2[J,K];

   vector[I+1] bias_std[J,K];
   vector[I+1] bias_std_bar[K];

   real<lower=linv_barp[3],upper=linv_barp[4]> linv_bar[K];
   real<lower=s2_barp[3],upper=s2_barp[4]> s2_bar[K];

}

transformed parameters
{

   vector[I+1] alpha[J,K];
   vector[I+1] beta[J,K];
   vector[I+1] bias[J,K];
   matrix[I+1,I+1] Sigma_bias[J,K];
   matrix[I+1,I+1] Chol_Sigma_bias[J,K];
   real xall[I+1];
   vector[I+1] eall[J,K];
   real s[J,K];
   real s_bar[K];
   vector[I+1] bias_bar[K];
   matrix[I+1,I+1] Sigma_bias_bar[K];
   matrix[I+1,I+1] Chol_Sigma_bias_bar[K];
   real l[J,K];
   real l_bar[K];
   
   s = sqrt(s2);
   s_bar = sqrt(s2_bar);
   l = inv(linv);
   l_bar = inv(linv_bar);

   for (i in 1:I) {
      xall[i] = x[i];
      for (k in 1:K) {
         for (j in 1:J) {
            eall[j,k,i] = e[j,k,i];
         }
      }
   }
   xall[I+1] = xhat;
   for (k in 1:K) {
      for (j in 1:J) {
         eall[j,k,I+1] = ehat[j,k];
      }
   }

   for (k in 1:K) {
      Sigma_bias_bar[k] = cov_exp_quad(xall, s_bar[k], l_bar[k]);
      for (i in 1:I+1) {
         Sigma_bias_bar[k,i,i] = Sigma_bias_bar[k,i,i] + 1e-12;
      }
      Chol_Sigma_bias_bar[k] = cholesky_decompose(Sigma_bias_bar[k]);
      bias_bar[k] = Chol_Sigma_bias_bar[k]*bias_std_bar[k];
      for (j in 1:J) {
         Sigma_bias[j,k] = cov_exp_quad(xall, s[j,k], l[j,k]);
         for (i in 1:I+1) {
            Sigma_bias[j,k,i,i] = Sigma_bias[j,k,i,i] + 1e-12;
         }
         Chol_Sigma_bias[j,k] = cholesky_decompose(Sigma_bias[j,k]);

         bias[j,k] = Chol_Sigma_bias[j,k]*bias_std[j,k]+bias_bar[k];

         for (i in 1:I+1) {
            alpha[j,k,i] = mu[j,k,i]*eall[j,k,i]*rho[j,k];
            beta[j,k,i] = (1-mu[j,k,i])*eall[j,k,i]*rho[j,k];
         }
      }
   }

}

model
{

   xhat ~ uniform(0,1);

   for (k in 1:K) {
      bias_std_bar[k] ~ normal(0,1);
      s2_bar[k] ~ cauchy(s2_barp[1],s2_barp[2]);
      linv_bar[k] ~ cauchy(linv_barp[1],linv_barp[2]);
      for (j in 1:J) {
         bias_std[j,k] ~ normal(0,1);
         s2[j,k] ~ cauchy(s2p[1],s2p[2]);
         linv[j,k] ~ cauchy(linvp[1],linvp[2]);
         rho[j,k] ~ gamma(rhop[1],rhop[2]);
         for (i in 1:I) {
            mu[j,k,i] ~ normal(normal_cdf(bias[j,k,i],0,sqrt(s2[j,k]+s2_bar[k])),mup[2]);
            m[j,k,i] ~ beta(alpha[j,k,i],beta[j,k,i]);
         }
         mhat[j,k] ~ beta(alpha[j,k,I+1],beta[j,k,I+1]);
         mu[j,k,I+1] ~ normal(normal_cdf(bias[j,k,I+1],0,sqrt(s2[j,k]+s2_bar[k])),mup[2]);
      }
   }

}

generated quantities {

   vector[I+1] mP[J,K];

   for (k in 1:K) {
      for (j in 1:J) {
         for (i in 1:I+1) {
            mP[j,k,i] = beta_rng(alpha[j,k,i],beta[j,k,i]);
         }
      }
   }

}
