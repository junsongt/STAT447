// data {
//   int<lower=0> n_above_limit;
//   int<lower=0> n_below_limit;
//   real<lower=0> limit;
//   vector<upper=limit>[n_below_limit] data_below_limit;
//   
// }
// 
// parameters {
//   real<lower=0> rate; 
//   vector<lower=limit>[n_above_limit] data_above_limit;
// }
// 
// model {
//   // prior
//   x ~ exponential(1.0/100);
//   
//   // likelihood
//   data_above_limit ~ exponential(x);
//   data_below_limit ~ exponential(x); 
// }
// 
// generated quantities {
//   real mean = 1.0/rate;
// }


// data {
//   int<lower=0> N;
//   real<lower=0> L; // limit
//   int<lower=0> n; // of obs < limit
//   vector<upper=L>[n] observed;
//   
// }
// 
// parameters {
//   real<lower=0> x;
//   vector<lower=L>[N-n] unobserved;
// }
// 
// model {
//   x ~ exponential(1.0/100);
//   observed ~ exponential(x);
//   unobserved ~ exponential(x);
// }
// 
// generated quantities {
//   real mean = 1.0/x;
// }


data {
  int<lower=0> N;
  real<lower=0> L; // limit
  int<lower=0> n; // of obs < limit
  vector<upper=L>[n] observed;
  
}

parameters {
  real<lower=0> x;
  vector<lower=0> h;
}



model {
  x ~ exponential(1.0/100);
  h ~ exponential(x);
  
}

generated quantities {
  real mean = 1.0/x;
}

