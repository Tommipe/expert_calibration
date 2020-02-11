data
{

   int I;               # Number of systems
   int J;               # Number of experts in a group
   int K;               # Number of groups

   vector[I] m[J,K];    # Experts' point estimates (calibration data)
   vector[I] x;         # Known parameter values (calibration data)

   real mhat[J,K];      # Experts' point estimates for the parameter in a new system

   real s2p[4];
   real linvp[4];
   real etap[4];
   real mup[4];
   real s2_mup[4];
  
}

parameters
{

   # unknown parameter value in a new system
   real<lower=0,upper=1> xhat;

   # parameters
   real<lower=etap[3],upper=etap[4]> eta[J,K];
   real<lower=linvp[3],upper=linvp[4]> linv[J,K];
   real<lower=s2p[3],upper=s2p[4]> s2[J,K];
   real<lower=s2_mup[3],upper=s2_mup[4]> s2_mu[J,K];
   vector<lower=mup[3],upper=mup[4]>[I+1] mu[J,K];

   vector[I+1] bias_std[J,K];

}

transformed parameters
{

   vector[I+1] alpha[J,K];
   vector[I+1] beta[J,K];
   vector[I+1] bias[J,K];
   matrix[I+1,I+1] Sigma_bias[J,K];
   matrix[I+1,I+1] Chol_Sigma_bias[J,K];
   real xall[I+1];
   real s[J,K];
   real l[J,K];
   vector[I+1] mu_mu[J,K];
   real s_mu[J,K];

   s = sqrt(s2);
   l = inv(linv);
   s_mu = sqrt(s2_mu);

   for (i in 1:I) {
      xall[i] = x[i];
   }
   xall[I+1] = xhat;

   for (k in 1:K) {
      for (j in 1:J) {
         Sigma_bias[j,k] = cov_exp_quad(xall, s[j,k], l[j,k]);
         for (i in 1:I+1) {
            Sigma_bias[j,k,i,i] = Sigma_bias[j,k,i,i] + 1e-12;
         }

         Chol_Sigma_bias[j,k] = cholesky_decompose(Sigma_bias[j,k]);
         bias[j,k] = Chol_Sigma_bias[j,k]*bias_std[j,k];

         for (i in 1:I+1) {
            mu_mu[j,k,i] = normal_cdf(bias[j,k,i],0,s[j,k]);
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