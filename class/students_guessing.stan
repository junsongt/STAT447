data {
  int n_students; 
  array[n_students] int<lower=0, upper=20> scores; 
}

parameters {
  real<lower=0, upper=1> fraction_guessing;
  real<lower=0, upper=1> non_guessing_population_mean;
  real<lower=0> non_guessing_population_spread;
  vector<lower=0, upper=1>[n_students] abilities;
}

transformed parameters {
  vector[n_students] complete_loglikelihood_guessing; 
  vector[n_students] complete_loglikelihood_non_guessing; 
  for (i in 1:n_students) {
    complete_loglikelihood_guessing[i]     
      = log(fraction_guessing) + binomial_lpmf(scores[i] | 20, 0.5); 
    complete_loglikelihood_non_guessing[i] 
      = log1p(-fraction_guessing) + binomial_lpmf(scores[i] | 20, abilities[i]);
  }
}

model {
  fraction_guessing ~ uniform(0, 1);
  non_guessing_population_mean ~ uniform(0, 1);
  non_guessing_population_spread ~ exponential(1.0/100);
  for (i in 1:n_students) {
    abilities[i] ~ beta_proportion(non_guessing_population_mean, non_guessing_population_spread);
    target += 
      log_sum_exp(complete_loglikelihood_guessing[i], complete_loglikelihood_non_guessing[i]);
  }
}

generated quantities {
  vector[n_students] guessing_probabilities = inv_logit(complete_loglikelihood_guessing - complete_loglikelihood_non_guessing);
  real predictive_score_non_guessing = 20 * beta_proportion_rng(non_guessing_population_mean, non_guessing_population_spread);
}
