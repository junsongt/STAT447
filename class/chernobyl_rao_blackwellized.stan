// Rao-Blackwellization
data {
  int<lower=0> N;
  real<lower=0> L; // limit
  int<lower=0> n; // number of obs below limit
  vector<upper=L>[n] observed;

}

parameters {
  real<lower=0> x;
}

model {
  x ~ exponential(1.0/100);
  observed ~ exponential(x);
  target += (N-n) * exponential_lccdf(L | x);

}

generated quantities {
  real mean = 1.0/x;
}


