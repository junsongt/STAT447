data {
  matrix[3,2] design_matrix; // number of successes
  vector[3] observations;
}

parameters {
  vector[2] coefficients;
}

model {
  coefficients[1] ~ cauchy(0, 1);
  coefficients[2] ~ cauchy(0, 1);
  
  for (i in 1:3) {
    observations[i] ~ normal(design_matrix[i] * coefficients, 1);
  }
}
