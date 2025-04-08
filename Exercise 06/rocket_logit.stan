data { 
  int N;
  array[N] int success;
  array[N] int launches;
}

parameters {
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  vector[N] slope;
  vector[N] intercept;
}

transformed parameters {
  vector[N] probs;
  for (i in 1:N) {
    vector<lower=0, upper=1> prob_i;
    for (j in 1:launches[i]) {
      prob_i[j] = inv_logit(slope*j + intercept);
    }
    probs[i] = prob_i;
  }
}

model {
  sigma1 ~ exponential(1.0/10000);
  sigma2 ~ exponential(1.0/10000);
  slope ~ normal(0, sigma1);
  intercept ~ normal(0, sigma2);
  for (i in 1:N) {
    for (j in 1:launches[i]) {
      target += bernoulli_lcdf(probs[i][j]);
    }
  }
  
}
