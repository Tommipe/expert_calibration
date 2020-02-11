data
{

   int N;               # Number of evaluation points for GP prediction
   vector[N] xP;        # Parameter values for GP prediction

   real s2p[4];
   real linvp[4];
   real etap[4];

}

parameters
{

   real<lower=linvp[3],upper=linvp[4]> linv;
   real<lower=s2p[3],upper=s2p[4]> s2;
   real<lower=etap[3],upper=etap[4]> eta;
   vector[N] bias_std;

}

transformed parameters
{

   vector[N] bias;
   vector<lower=0,upper=1>[N] mu;
   matrix[N,N] Sigma_bias;
   matrix[N,N] Chol_Sigma_bias;
   real xall[N];
   real s;
   real l;

   s = sqrt(s2);
   l = inv(linv);

   for (n in 1:N) {
      xall[n] = xP[n];
   }

   Sigma_bias = cov_exp_quad(xall, s, l);
   for (in in 1:N) {
      Sigma_bias[in,in] = Sigma_bias[in,in] + 1e-12;
   }

   Chol_Sigma_bias = cholesky_decompose(Sigma_bias);
   bias = Chol_Sigma_bias*bias_std;

   for (n in 1:N) {
      mu[n] = normal_cdf(bias[n],0,s);
   }

}

model
{

   bias_std ~ normal(0,1);
   s2 ~ cauchy(s2p[1],s2p[2]);
   linv ~ cauchy(linvp[1],linvp[2]);

   eta ~ gamma(etap[1],etap[2]);

}

generated quantities
{

   vector<lower=0,upper=1>[N] yP;
   vector<lower=0>[N] alpha;
   vector<lower=0>[N] beta;

   alpha = mu*eta;
   beta = (1-mu)*eta;

   for (n in 1:N) {
      yP[n] = beta_rng(alpha[n],beta[n]);
   }

}