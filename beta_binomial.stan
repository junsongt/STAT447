data {
  int<lower=0> n;         // number of trials
  int<lower=0,upper=n> k; // number of successes
}

parameters {
  real<lower=0,upper=1> p;
}

model {
  // prior
  p ~ beta(1,1);

  // likelihood
  k ~ binomial(n, p);
}
