data
{

   int I;               # Number of systems

   vector[I] m;         # Expert's point estimates (calibration data)
   vector[I] x;         # Known parameter values (calibration data)

   real mhat;           # Expert's point estimate for the parameter in a new system

   real s2p[4];
   real linvp[4];
   real etap[4];
  
}

parameters
{

   # unknown parameter value in a new system
   real<lower=0,upper=1> xhat;

   # parameters
   real<lower=etap[3],upper=etap[4]> eta;
   real<lower=linvp[3],upper=linvp[4]> linv;
   real<lower=s2p[3],upper=s2p[4]> s2;

   vector[I+1] bias_std;

}

transformed parameters
{

   vector<lower=0,upper=1>[I+1] mu;
   vector[I+1] alpha;
   vector[I+1] beta;
   vector[I+1] bias;
   matrix[I+1,I+1] Sigma_bias;
   matrix[I+1,I+1] Chol_Sigma_bias;
   real xall[I+1];
   real s;
   real l;

   s = sqrt(s2);
   l = inv(linv);

   for (i in 1:I) {
      xall[i] = x[i];
   }
   xall[I+1] = xhat;

   Sigma_bias = cov_exp_quad(xall, s, l);
   for (i in 1:I+1) {
      Sigma_bias[i,i] = Sigma_bias[i,i] + 1e-12;
   }

   Chol_Sigma_bias = cholesky_decompose(Sigma_bias);
   bias = Chol_Sigma_bias*bias_std;

   for (i in 1:I+1) {
      mu[i] = normal_cdf(bias[i],0,s);
      alpha[i] = mu[i]*eta;
      beta[i] = (1-mu[i])*eta;
   }

}

model
{

   xhat ~ uniform(0,1);

   bias_std ~ normal(0,1);
   s2 ~ cauchy(s2p[1],s2p[2]);
   linv ~ cauchy(linvp[1],linvp[2]);

   eta ~ gamma(etap[1],etap[2]);

   for (i in 1:I) {
      m[i] ~ beta(alpha[i],beta[i]);
   }
   mhat ~ beta(alpha[I+1],beta[I+1]);

}