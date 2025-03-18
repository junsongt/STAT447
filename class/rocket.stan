data { 
  int N; 
  array[N] int y; 
}

parameters { 
  real slope;
  real intercept;
}

transformed parameters {
  vector[N] theta = inv_logit(intercept + slope*linspaced_vector(N,1,N));
  // int<lower=0, upper=1> pos_slope = slope > 0; // int type can't be in transformed parameters
  // real<lower=0, upper=1> pos_slope = slope > 0;
}

model {
  slope ~ normal(0, 10);
  intercept ~ normal(0, 10);
  y ~ bernoulli(theta);
  
}

generated quantities {
  int<lower=0, upper=1> pos_slope = slope > 0;
}

