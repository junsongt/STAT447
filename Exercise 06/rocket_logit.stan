data { 
  int<lower=0> N; // number of types of rocket
  // array[N] int launch_res;
  int<lower=0> launches[N];
  int<upper=1> launch_res[N, max(launches)];
  // matrix<lower=0, upper=1>[N, max(launches)] launch_res;
}

parameters {
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  vector[N] slope;
  vector[N] intercept;
}

transformed parameters {
  matrix<lower=0, upper=1>[N, max(launches)] probs;
  for (i in 1:N) {
    for (j in 1:max(launches)) {
      probs[i,j] = (launches[i] <= max(launches)) ? inv_logit(slope*j + intercept) : 1;
    }
  }
}

model {
  sigma1 ~ exponential(1.0/10000);
  sigma2 ~ exponential(1.0/10000);
  slope ~ normal(0, sigma1);
  intercept ~ normal(0, sigma2);
  for (i in 1:N) {
    for (j in 1:launches[i]) {
      target += (launches[i] <= max(launches)) ? bernoulli_lpmf(launch_res[i,j] | probs[i,j]) : 0;
    }
  }
}

