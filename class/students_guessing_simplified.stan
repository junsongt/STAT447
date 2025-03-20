// Alex's version
data {
  int<lower=0, upper=20> score;
}

parameters {
  real<lower=0, upper=1> ability;
}

transformed parameters {
  real complete_likelihood_guessing
    = 1.0/3 * exp(binomial_lpmf(score | 20, 0.5));
  real complete_likelihood_non_guessing
    = 2.0/3 * exp(binomial_lpmf(score | 20, ability));
}

model {
  ability ~ uniform(0, 1);
  target +=
    log(complete_likelihood_guessing + complete_likelihood_non_guessing);
}

generated quantities {
  real guessing_probability =
    complete_likelihood_guessing / (complete_likelihood_guessing + complete_likelihood_non_guessing);
}


// // my version
// data {
//   int<lower=0, upper=20> score;
// }
// 
// parameters {
//   real<lower=0, upper=1> ability;
//   real<lower=0, upper=1> g;
// }
// 
// transformed parameters {
//   real complete_likelihood_guessing
//     = 1.0/3 * exp(binomial_lpmf(score | 20, 0.5));
//   real complete_likelihood_non_guessing
//     = 2.0/3 * exp(binomial_lpmf(score | 20, ability));
//   real g_hat = g <= 1.0/3? 1 : 0;
//   real prob = 0.5 * g_hat + ability * (1 - g_hat);
// }
// 
// model {
//   ability ~ uniform(0, 1);
//   g ~ uniform(0,1);
//   score ~ binomial(20, prob);
//   // target +=
//   //   log(complete_likelihood_guessing + complete_likelihood_non_guessing);
// }
// 
// generated quantities {
//   real guessing_probability =
//     complete_likelihood_guessing / (complete_likelihood_guessing + complete_likelihood_non_guessing);
// }
