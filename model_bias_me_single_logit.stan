data
{

   int N;               # Number of evaluation points for GP prediction
   vector[N] xP;        # Parameter values for GP prediction
   int I;               # Number of calibration data points
   vector[I] x;        # True parameter values
   vector[I] m;        # Expert's parameter values
   vector[I] e;       # Expert's uncertainty estimate

   real s2p[4];
   real linvp[4];
   real rhop[4];

}

parameters
{

   real<lower=linvp[3],upper=linvp[4]> linv;
   real<lower=s2p[3],upper=s2p[4]> s2;
   real<lower=rhop[3],upper=rhop[4]> rho;
   vector[I+N] bias_std;

}

transformed parameters
{

   vector<lower=0,upper=1>[I] mu;
   vector[I] alpha;
   vector[I] beta;
   vector[I+N] bias;
   matrix[I+N,I+N] Sigma_bias;
   matrix[I+N,I+N] Chol_Sigma_bias;
   real xall[I+N];
   real s;
   real l;

   s = sqrt(s2);
   l = inv(linv);

   for (i in 1:I) {
      xall[i] = x[i];
   }
   for (n in 1:N) {
      xall[I+n] = xP[n];
   }

   Sigma_bias = cov_exp_quad(xall, s, l);
   for (in in 1:I+N) {
      Sigma_bias[in,in] = Sigma_bias[in,in] + 1e-12;
   }

   Chol_Sigma_bias = cholesky_decompose(Sigma_bias);
   bias = Chol_Sigma_bias*bias_std;

   for (i in 1:I) {
      mu[i] = inv_logit(bias[i]+logit(x[i]));
      alpha[i] = mu[i]*e[i]*rho;
      beta[i] = (1-mu[i])*e[i]*rho;
   }

}

model
{

   bias_std ~ normal(0,1);
   s2 ~ cauchy(s2p[1],s2p[2]);
   linv ~ cauchy(linvp[1],linvp[2]);

   rho ~ gamma(rhop[1],rhop[2]);

   for (i in 1:I) {
      m[i] ~ beta(alpha[i],beta[i]);
   }

}