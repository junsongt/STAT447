data { 
  int N; 
  array[N] int failures;
  array[N] int launches;
}

parameters { 
  real<lower=0> mu;
  real<lower=0> s;
  vector<lower=0, upper=1>[N] probs; 
}


model {
  mu ~ uniform(0,1);
  s ~ exponential(1.0/10000);
  probs ~ beta_proportion(mu, s);
  failures ~ binomial(launches, probs);
}

// generated quantities {
//   int<lower=0, upper=1> pred[N];
//   for (i in 1:N) {
//     pred[i] = bernoulli_rng(probs[i])
//   }
// }


