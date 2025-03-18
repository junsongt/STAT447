data {
  int<lower=0> N;
  vector<lower=0, upper=1>[N] y;
  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real<lower=0> theta0;
  real theta1;
  real theta2;
}

transformed parameters {
  vector<lower=0, upper=1>[N] mu;
  for (i in 1:N) {
    mu[i] = inv_logit(theta1*(i/N)+theta2);
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  theta0 ~ exponential(1);
  theta1 ~ normal(0, 1000);
  theta2 ~ normal(0, 1000);
  for (i in 1:N) {
    y[i] ~ beta_proportion(mu[i], theta0);
  }
}

generated quantities {
  real<lower=0, upper=1> mu_pred = inv_logit(theta1*((N+1)/N) + theta2);
  real y_pred = beta_proportion_rng(mu_pred, theta0);
}



