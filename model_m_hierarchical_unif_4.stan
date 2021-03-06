data
{

   int I;               # Number of systems
   int J;               # Number of experts in a group
   int K;               # Number of groups
   int N;               # Number of points for plotting

   vector[I] m[J,K];    # Experts' point estimates (calibration data)
   vector[I] x;         # Known parameter values (calibration data)
   vector[N] xP;        # Points for plotting

   real mhat[J,K];      # Experts' point estimates for the parameter in a new system

   real s2p[4];
   real linvp[4];
   real etap[4];
   real s2_barp[4];
   real linv_barp[4];
   real mup[4];
   real s2_mup[4];
  
}

parameters
{

   # parameters
   real<lower=etap[3],upper=etap[4]> eta[J,K];
   real<lower=linvp[3],upper=linvp[4]> linv[J,K];
   real<lower=s2p[3],upper=s2p[4]> s2[J,K];
   real<lower=s2_mup[3],upper=s2_mup[4]> s2_mu[J,K];
   vector<lower=mup[3],upper=mup[4]>[I+1] mu[J,K];

   vector[I+1] bias_std[J,K];
   vector[I+1] bias_std_bar[K];

   real<lower=linv_barp[3],upper=linv_barp[4]> linv_bar[K];
   real<lower=s2_barp[3],upper=s2_barp[4]> s2_bar[K];


}

transformed parameters
{

   vector[I+N] alpha[J,K];
   vector[I+N] beta[J,K];
   vector[I+N] bias[J,K];
   matrix[I+N,I+N] Sigma_bias[J,K];
   matrix[I+N,I+N] Chol_Sigma_bias[J,K];
   real xall[I+N];
   real s[J,K];
   real l[J,K];
   real s_bar[K];
   real l_bar[K];
   vector[I+N] bias_bar[K];
   matrix[I+N,I+N] Sigma_bias_bar[K];
   matrix[I+N,I+N] Chol_Sigma_bias_bar[K];
   vector[I+N] mu_mu[J,K];
   real s_mu[J,K];

   s = sqrt(s2);
   l = inv(linv);
   s_bar = sqrt(s2_bar);
   l_bar = inv(linv_bar);
   s_mu = sqrt(s2_mu);

   for (i in 1:I) {
      xall[i] = x[i];
   }
   for (n in 1:N) {
      xall[I+n] = xP[n];
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
            mu_mu[j,k,i] = normal_cdf(bias[j,k,i],0,sqrt(s2[j,k]+s2_bar[k]));
            alpha[j,k,i] = mu[j,k,i]*eta[j,k];
            beta[j,k,i] = (1-mu[j,k,i])*eta[j,k];
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
         eta[j,k] ~ gamma(etap[1],etap[2]);
         s2_mu[j,k] ~ cauchy(s2_mup[1],s2_mup[2]);
         for (i in 1:I) {
            mu[j,k,i] ~ normal(mu_mu[j,k,i],s_mu[j,k]);
            m[j,k,i] ~ beta(alpha[j,k,i],beta[j,k,i]);
         }
         mhat[j,k] ~ beta(alpha[j,k,I+1],beta[j,k,I+1]);
         mu[j,k,I+1] ~ normal(mu_mu[j,k,I+1],s_mu[j,k]);
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