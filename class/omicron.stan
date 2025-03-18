data {
  int N;
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
  // vector[N] mu = inv_logit(theta2 + theta1*linspaced_vector(N,0,1));
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = inv_logit(theta2 + theta1 * (i/(1.0*N)));
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  theta0 ~ exponential(1);
  theta1 ~ normal(0, 1000);
  theta2 ~ normal(0, 1000);
  // using loop
  // for (i in 1:N) {
  //   y[i] ~ beta_proportion(inv_logit(theta2 + theta1 * (i/N)), theta0);
  // }
  for (i in 1:N) {
    y[i] ~ beta_proportion(mu[i], theta0);
  }
  // y ~ beta_proportion(mu, theta0);
}

generated quantities {
  real mu_pred = inv_logit(theta1*((N+1)/N) + theta2);
  real y_pred = beta_proportion_rng(mu_pred, theta0);
}



