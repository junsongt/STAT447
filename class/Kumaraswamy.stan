// data {
//   real<lower=0> a;
//   real<lower=0> b;  
// }
// 
// parameters {
//   real<lower=0, upper=1> x;
// }
// 
// model {
//   target += log(a) + log(b) + (a-1) * log(x) + (b-1) * log1p(-x^a);
//                                                     // ^ log1p(z) = log(1+z)
// }

// For Stan to pick up your function in its ~ syntax, 
// you must use the special suffix _lpdf for your function name, 
// which stands for “log probability density function.”
functions {
  real Kumaraswamy_lpdf(real x, real a, real b) {
    return log(a) + log(b) + (a-1) * log(x) + (b-1) * log1p(-x^a);
  }
}

data {
  real<lower=0> a;
  real<lower=0> b;  
}

parameters {
  real<lower=0, upper=1> x; 
}

model {
  x ~ Kumaraswamy(a, b);
}
