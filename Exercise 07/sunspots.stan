data {
    int<lower=0> N;
    int<lower=0> y[N];
    vector<lower=0>[N] t;
    // vector<lower=0>[N] y; ONLY for real array
}

parameters {
    real<lower=0, upper=200> theta1;
    real<lower=0, upper=10> theta2;
    real<lower=0, upper=2*pi()> theta3;
}

model {
    // prior
    theta1 ~ uniform(0, 200);
    theta2 ~ uniform(0, 10);
    theta3 ~ uniform(0, 2*pi());

    // likelihood
    // y ~ poisson(theta1 * (sin(theta2*linspaced_vector(N,1,N) + theta3) + 1.1));
    y ~ poisson(theta1 * (sin(theta2*t + theta3) + 1.1));
}

